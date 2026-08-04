// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import SebbuFFT
import Testing

@Suite("Portable transform plans")
struct PortableTransformTests {
    @Test("Forward transform matches the naive DFT", arguments: transformSampleCounts)
    func forwardMatchesNaiveDFT(sampleCount: Int) {
        let input = deterministicSignal(count: sampleCount, seed: 1)
        let plan = FFT.FFTPlan(sampleSize: sampleCount)
        expectForwardPlanMatchesOracle(plan, input: input, spacing: 0.125)
    }

    @Test("Inverse transform matches the naive DFT", arguments: transformSampleCounts)
    func inverseMatchesNaiveDFT(sampleCount: Int) {
        let input = deterministicSignal(count: sampleCount, seed: 2)
        let plan = FFT.iFFTPlan(sampleSize: sampleCount)
        expectInversePlanMatchesOracle(plan, input: input, spacing: 0.375)
    }

    @Test("An impulse has a flat spectrum", arguments: [3, 5, 8, 11, 16])
    func impulseHasFlatSpectrum(sampleCount: Int) {
        var input = [Complex<Double>](repeating: .zero, count: sampleCount)
        input[0] = Complex(1.0, 0.0)

        let (_, spectrum) = FFT.FFTPlan(sampleSize: sampleCount)
            .execute(input, spacing: 1.0)
        let expected = [Complex<Double>](
            repeating: Complex(1.0, 0.0),
            count: sampleCount
        )

        expectApproximatelyEqual(spectrum, expected)
    }

    @Test("A constant signal occupies only the DC bin", arguments: [3, 5, 8, 11, 16])
    func constantSignalOccupiesDCBin(sampleCount: Int) {
        let value = Complex(1.25, -0.75)
        let input = [Complex<Double>](repeating: value, count: sampleCount)
        let (_, spectrum) = FFT.FFTPlan(sampleSize: sampleCount)
            .execute(input, spacing: 1.0)

        var expected = [Complex<Double>](repeating: .zero, count: sampleCount)
        expected[0] = Complex(
            value.real * Double(sampleCount),
            value.imaginary * Double(sampleCount)
        )
        expectApproximatelyEqual(spectrum, expected)
    }

    @Test("A complex tone occupies its corresponding bin", arguments: [5, 7, 8, 11, 16])
    func complexToneOccupiesOneBin(sampleCount: Int) {
        let bin = 2
        let input = (0..<sampleCount).map { index in
            Complex<Double>(
                length: 1.0,
                phase: 2.0 * Double.pi * Double(bin * index) / Double(sampleCount)
            )
        }

        let (_, spectrum) = FFT.FFTPlan(sampleSize: sampleCount)
            .execute(input, spacing: 1.0)
        var expected = [Complex<Double>](repeating: .zero, count: sampleCount)
        expected[bin] = Complex(Double(sampleCount), 0.0)

        expectApproximatelyEqual(spectrum, expected)
    }

    @Test("Forward and inverse plans round-trip values and spacing", arguments: transformSampleCounts)
    func roundTrip(sampleCount: Int) {
        let input = deterministicSignal(count: sampleCount, seed: 3)
        let sampleSpacing = 0.0625
        let angularFrequencySpacing = 2.0 * Double.pi
            / (Double(sampleCount) * sampleSpacing)

        let (_, spectrum) = FFT.FFTPlan(sampleSize: sampleCount)
            .execute(input, spacing: sampleSpacing)
        let (samplePositions, reconstructed) = FFT.iFFTPlan(sampleSize: sampleCount)
            .execute(spectrum, spacing: angularFrequencySpacing)

        let expectedPositions = (0..<sampleCount).map {
            Double($0) * sampleSpacing
        }
        expectApproximatelyEqual(samplePositions, expectedPositions)
        expectApproximatelyEqual(reconstructed, input)
    }

    @Test("The forward transform is linear", arguments: [3, 5, 8, 11, 16])
    func forwardTransformIsLinear(sampleCount: Int) {
        let first = deterministicSignal(count: sampleCount, seed: 4)
        let second = deterministicSignal(count: sampleCount, seed: 5)
        let firstScale = Complex(0.75, -0.25)
        let secondScale = Complex(-0.5, 0.625)
        let combined = zip(first, second).map {
            firstScale * $0.0 + secondScale * $0.1
        }

        let plan = FFT.FFTPlan(sampleSize: sampleCount)
        let (_, combinedTransform) = plan.execute(combined, spacing: 1.0)
        let (_, firstTransform) = plan.execute(first, spacing: 1.0)
        let (_, secondTransform) = plan.execute(second, spacing: 1.0)
        let expected = zip(firstTransform, secondTransform).map {
            firstScale * $0.0 + secondScale * $0.1
        }

        expectApproximatelyEqual(combinedTransform, expected)
    }

    @Test("Parseval identity uses the documented normalization", arguments: [3, 5, 8, 11, 16])
    func parsevalIdentity(sampleCount: Int) {
        let input = deterministicSignal(count: sampleCount, seed: 6)
        let (_, spectrum) = FFT.FFTPlan(sampleSize: sampleCount)
            .execute(input, spacing: 1.0)

        let timeEnergy = input.reduce(0.0) { $0 + magnitudeSquared($1) }
        let frequencyEnergy = spectrum.reduce(0.0) {
            $0 + magnitudeSquared($1)
        } / Double(sampleCount)
        let scale = max(1.0, max(timeEnergy, frequencyEnergy))

        #expect(abs(timeEnergy - frequencyEnergy) <= 5e-11 * scale)
    }

    @Test("Real inputs have conjugate-symmetric spectra", arguments: [3, 4, 5, 8, 11, 16])
    func realInputHasConjugateSymmetry(sampleCount: Int) {
        let input = deterministicSignal(count: sampleCount, seed: 7).map {
            Complex($0.real, 0.0)
        }
        let (_, spectrum) = FFT.FFTPlan(sampleSize: sampleCount)
            .execute(input, spacing: 1.0)

        for index in 1..<sampleCount {
            let mirroredIndex = (sampleCount - index) % sampleCount
            expectApproximatelyEqual(
                spectrum[mirroredIndex],
                spectrum[index].conjugate,
                context: "bins \(index) and \(mirroredIndex):"
            )
        }
    }

    @Test("A plan can be reused without retaining transform state")
    func planReuse() {
        let sampleCount = 11
        let plan = FFT.FFTPlan(sampleSize: sampleCount)

        for seed in 0..<8 {
            let input = deterministicSignal(count: sampleCount, seed: seed)
            expectForwardPlanMatchesOracle(plan, input: input, spacing: 0.25)
        }
    }

    @Test("Empty portable transforms return empty outputs")
    func emptyInput() {
        let forward = FFT.FFTPlan(sampleSize: 0).execute([], spacing: 1.0)
        let inverse = FFT.iFFTPlan(sampleSize: 0).execute([], spacing: 1.0)

        #expect(forward.x.isEmpty)
        #expect(forward.y.isEmpty)
        #expect(inverse.x.isEmpty)
        #expect(inverse.y.isEmpty)
    }
}

@Suite("Default plan factories")
struct DefaultPlanTests {
    @Test("The default forward plan follows the package convention", arguments: [8, 11, 16])
    func defaultForwardMatchesOracle(sampleCount: Int) {
        let input = deterministicSignal(count: sampleCount, seed: 8)
        let plan = FFT.defaultFFTPlan(sampleSize: sampleCount)
        expectForwardPlanMatchesOracle(plan, input: input, spacing: 0.125)
    }

    @Test("The default inverse plan follows the package convention", arguments: [8, 11, 16])
    func defaultInverseMatchesOracle(sampleCount: Int) {
        let input = deterministicSignal(count: sampleCount, seed: 9)
        let plan = FFT.defaultiFFTPlan(sampleSize: sampleCount)
        expectInversePlanMatchesOracle(plan, input: input, spacing: 0.375)
    }
}

@Suite("One-shot FFT API")
struct OneShotAPITests {
    @Test("FFT.fft matches the documented forward transform", arguments: [1, 5, 8, 11, 16])
    func forwardTransform(sampleCount: Int) {
        let input = deterministicSignal(count: sampleCount, seed: 18)
        let spacing = 0.125
        let result = FFT.fft(input, spacing: spacing)

        expectApproximatelyEqual(
            result.frequencies,
            expectedForwardAxis(count: sampleCount, spacing: spacing)
        )
        expectApproximatelyEqual(result.spectrum, naiveDFT(input))
    }

    @Test("FFT.ifft matches the documented inverse transform", arguments: [1, 5, 8, 11, 16])
    func inverseTransform(sampleCount: Int) {
        let input = deterministicSignal(count: sampleCount, seed: 19)
        let spacing = 0.375
        let result = FFT.ifft(input, spacing: spacing)

        expectApproximatelyEqual(
            result.t,
            expectedInverseAxis(count: sampleCount, spacing: spacing)
        )
        expectApproximatelyEqual(result.signal, naiveDFT(input, inverse: true))
    }

    @Test("Empty one-shot transforms return empty outputs")
    func emptyInput() {
        let forward = FFT.fft([])
        let inverse = FFT.ifft([])

        #expect(forward.frequencies.isEmpty)
        #expect(forward.spectrum.isEmpty)
        #expect(inverse.t.isEmpty)
        #expect(inverse.signal.isEmpty)
    }
}