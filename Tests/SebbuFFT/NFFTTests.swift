// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import RealModule
import SebbuFFT
import Testing

private let nonuniformPositions: [Double] = [
    -0.499,
    -0.431,
    -0.307,
    -0.123,
    -0.001,
    0.071,
    0.249,
    0.413,
    0.499,
]

private func nfftCoefficients(count: Int, seed: Int) -> [Complex<Double>] {
    precondition(count.isMultiple(of: 2))
    return deterministicSignal(count: count, seed: seed)
}

private func complexInnerProduct(
    _ lhs: [Complex<Double>],
    _ rhs: [Complex<Double>]
) -> Complex<Double> {
    precondition(lhs.count == rhs.count)
    return zip(lhs, rhs).reduce(into: Complex<Double>.zero) { result, pair in
        result += pair.0.conjugate * pair.1
    }
}

@Suite("Direct non-equispaced transforms")
struct DirectNFFTTests {
    @Test("A single Fourier mode has the expected phase")
    func singleMode() {
        let coefficientCount = 8
        let waveNumber = 2
        var coefficients = [Complex<Double>](
            repeating: .zero,
            count: coefficientCount
        )
        coefficients[coefficientCount / 2 + waveNumber] = Complex(1.0, 0.0)

        let actual = FFT.ndft(coefficients, at: nonuniformPositions)
        let expected = nonuniformPositions.map { position in
            Complex<Double>(
                length: 1.0,
                phase: -2.0 * Double.pi * Double(waveNumber) * position
            )
        }

        expectApproximatelyEqual(actual, expected)
    }

    @Test("The direct adjoint satisfies the inner-product identity")
    func directAdjointIdentity() {
        let coefficients = nfftCoefficients(count: 16, seed: 31)
        let values = deterministicSignal(
            count: nonuniformPositions.count,
            seed: 32
        )

        let transformed = FFT.ndft(coefficients, at: nonuniformPositions)
        let adjoint = FFT.ndftAdjoint(
            values,
            at: nonuniformPositions,
            coefficientCount: coefficients.count
        )

        expectApproximatelyEqual(
            complexInnerProduct(transformed, values),
            complexInnerProduct(coefficients, adjoint),
            absoluteTolerance: 5e-11,
            relativeTolerance: 5e-11
        )
    }

    @Test("Positions are periodic with period one")
    func positionsArePeriodic() {
        let coefficients = nfftCoefficients(count: 16, seed: 33)
        let shiftedPositions = nonuniformPositions.enumerated().map {
            $0.element + Double($0.offset - 4)
        }

        expectApproximatelyEqual(
            FFT.ndft(coefficients, at: shiftedPositions),
            FFT.ndft(coefficients, at: nonuniformPositions)
        )
    }
}

@Suite("Non-equispaced fast transforms")
struct NFFTTests {
    @Test(
        "The forward NFFT matches the direct transform",
        arguments: [8, 16, 30, 50]
    )
    func forwardMatchesDirect(coefficientCount: Int) {
        let coefficients = nfftCoefficients(
            count: coefficientCount,
            seed: coefficientCount
        )
        let configuration = FFT.NFFTConfiguration(tolerance: 1e-10)

        let expected = FFT.ndft(coefficients, at: nonuniformPositions)
        let actual = FFT.nfft(
            coefficients,
            at: nonuniformPositions,
            configuration: configuration
        )

        expectApproximatelyEqual(
            actual,
            expected,
            absoluteTolerance: 1e-8,
            relativeTolerance: 1e-10
        )
    }

    @Test(
        "The adjoint NFFT matches the direct adjoint",
        arguments: [8, 16, 30, 50]
    )
    func adjointMatchesDirect(coefficientCount: Int) {
        let values = deterministicSignal(
            count: nonuniformPositions.count,
            seed: coefficientCount + 1
        )
        let configuration = FFT.NFFTConfiguration(tolerance: 1e-10)

        let expected = FFT.ndftAdjoint(
            values,
            at: nonuniformPositions,
            coefficientCount: coefficientCount
        )
        let actual = FFT.nfftAdjoint(
            values,
            at: nonuniformPositions,
            coefficientCount: coefficientCount,
            configuration: configuration
        )

        expectApproximatelyEqual(
            actual,
            expected,
            absoluteTolerance: 1e-8,
            relativeTolerance: 1e-10
        )
    }

    @Test("Fast forward and adjoint plans satisfy the inner-product identity")
    func fastAdjointIdentity() {
        let coefficientCount = 16
        let configuration = FFT.NFFTConfiguration(tolerance: 1e-10)
        let gridSize = coefficientCount * configuration.oversamplingFactor
        let forwardPlan = FFT.NFFTPlan(
            positions: nonuniformPositions,
            coefficientCount: coefficientCount,
            configuration: configuration,
            fftPlan: FFT.FFTPlan(sampleSize: gridSize)
        )
        let adjointPlan = FFT.NFFTAdjointPlan(
            positions: nonuniformPositions,
            coefficientCount: coefficientCount,
            configuration: configuration,
            inverseFFTPlan: FFT.iFFTPlan(sampleSize: gridSize)
        )
        let coefficients = nfftCoefficients(count: coefficientCount, seed: 35)
        let values = deterministicSignal(
            count: nonuniformPositions.count,
            seed: 36
        )

        let lhs = complexInnerProduct(forwardPlan.execute(coefficients), values)
        let rhs = complexInnerProduct(coefficients, adjointPlan.execute(values))

        expectApproximatelyEqual(
            lhs,
            rhs,
            absoluteTolerance: 2e-10,
            relativeTolerance: 2e-10
        )
    }

    @Test("A plan caches its shape and can be reused")
    func planReuse() {
        let coefficientCount = 16
        let configuration = FFT.NFFTConfiguration(
            oversamplingFactor: 4,
            tolerance: 1e-8,
            windowHalfWidth: 7
        )
        let plan = FFT.NFFTPlan(
            positions: nonuniformPositions,
            coefficientCount: coefficientCount,
            configuration: configuration
        )

        #expect(plan.coefficientCount == coefficientCount)
        #expect(plan.positionCount == nonuniformPositions.count)
        #expect(plan.gridSize == coefficientCount * 4)
        #expect(plan.windowHalfWidth == 7)

        for seed in 40..<44 {
            let coefficients = nfftCoefficients(
                count: coefficientCount,
                seed: seed
            )
            expectApproximatelyEqual(
                plan.execute(coefficients),
                FFT.nfft(
                    coefficients,
                    at: nonuniformPositions,
                    configuration: configuration
                ),
                absoluteTolerance: 5e-11,
                relativeTolerance: 5e-11
            )
        }
    }

    @Test("Fast transforms preserve periodic position wrapping")
    func positionsArePeriodic() {
        let coefficients = nfftCoefficients(count: 16, seed: 45)
        let shiftedPositions = nonuniformPositions.enumerated().map {
            $0.element + Double($0.offset - 4)
        }

        expectApproximatelyEqual(
            FFT.nfft(coefficients, at: shiftedPositions),
            FFT.nfft(coefficients, at: nonuniformPositions),
            absoluteTolerance: 5e-11,
            relativeTolerance: 5e-11
        )
    }

    @Test("Empty position sets produce correctly shaped results")
    func emptyPositions() {
        let coefficientCount = 16
        let coefficients = nfftCoefficients(
            count: coefficientCount,
            seed: 46
        )
        let forwardPlan = FFT.NFFTPlan(
            positions: [],
            coefficientCount: coefficientCount
        )
        let adjointPlan = FFT.NFFTAdjointPlan(
            positions: [],
            coefficientCount: coefficientCount
        )

        #expect(forwardPlan.execute(coefficients).isEmpty)
        expectApproximatelyEqual(
            adjointPlan.execute([]),
            [Complex<Double>](repeating: .zero, count: coefficientCount)
        )
    }
}

@Suite("Gaussian NFFT kernel")
struct GaussianNFFTKernelTests {
    @Test("Estimated half-width meets the analytic error target")
    func estimatedWindowHalfWidth() {
        let kernel = FFT.GaussianNFFTKernel()
        let tolerance = 1e-10
        let coefficientCount = 64
        let oversamplingFactor = 3
        let halfWidth = kernel.estimateWindowHalfWidth(
            tolerance: tolerance,
            coefficientCount: coefficientCount,
            oversamplingFactor: oversamplingFactor
        )

        #expect(halfWidth > 0)
        #expect(
            kernel.errorConstant(
                windowHalfWidth: halfWidth,
                oversamplingFactor: oversamplingFactor
            ) <= tolerance / Double(coefficientCount)
        )
    }
}
