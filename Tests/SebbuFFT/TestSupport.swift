import ComplexModule
import SebbuFFT
import Testing

let transformSampleCounts = [1, 2, 3, 4, 5, 7, 8, 11, 16]

func complexMagnitude(_ value: Complex<Double>) -> Double {
    (value.real * value.real + value.imaginary * value.imaginary).squareRoot()
}

func magnitudeSquared(_ value: Complex<Double>) -> Double {
    value.real * value.real + value.imaginary * value.imaginary
}

func expectApproximatelyEqual(
    _ actual: Complex<Double>,
    _ expected: Complex<Double>,
    absoluteTolerance: Double = 2e-11,
    relativeTolerance: Double = 2e-11,
    context: String = ""
) {
    let error = complexMagnitude(actual - expected)
    let scale = max(1.0, max(complexMagnitude(actual), complexMagnitude(expected)))
    let tolerance = absoluteTolerance + relativeTolerance * scale

    #expect(
        error <= tolerance,
        "\(context) expected \(expected), got \(actual); error \(error), tolerance \(tolerance)"
    )
}

func expectApproximatelyEqual(
    _ actual: [Complex<Double>],
    _ expected: [Complex<Double>],
    absoluteTolerance: Double = 2e-11,
    relativeTolerance: Double = 2e-11
) {
    #expect(actual.count == expected.count)
    guard actual.count == expected.count else { return }

    for index in actual.indices {
        expectApproximatelyEqual(
            actual[index],
            expected[index],
            absoluteTolerance: absoluteTolerance,
            relativeTolerance: relativeTolerance,
            context: "index \(index):"
        )
    }
}

func expectApproximatelyEqual(
    _ actual: [Double],
    _ expected: [Double],
    absoluteTolerance: Double = 2e-12,
    relativeTolerance: Double = 2e-12
) {
    #expect(actual.count == expected.count)
    guard actual.count == expected.count else { return }

    for index in actual.indices {
        let scale = max(1.0, max(abs(actual[index]), abs(expected[index])))
        let tolerance = absoluteTolerance + relativeTolerance * scale
        #expect(
            abs(actual[index] - expected[index]) <= tolerance,
            "index \(index): expected \(expected[index]), got \(actual[index])"
        )
    }
}

func deterministicSignal(count: Int, seed: Int = 0) -> [Complex<Double>] {
    (0..<count).map { index in
        let realInteger = (index * 17 + seed * 5 + 3) % 23 - 11
        let imaginaryInteger = (index * 7 + seed * 11 + 1) % 19 - 9
        let real = Double(realInteger) / 7.0 + Double(index) / 31.0
        let imaginary = Double(imaginaryInteger) / 5.0 - Double(index) / 29.0
        return Complex(real, imaginary)
    }
}

/// A deliberately direct O(N^2) implementation used as an independent oracle.
func naiveDFT(
    _ input: [Complex<Double>],
    inverse: Bool = false
) -> [Complex<Double>] {
    let count = input.count
    guard count > 0 else { return [] }

    let sign = inverse ? 1.0 : -1.0
    let scale = inverse ? 1.0 / Double(count) : 1.0

    return (0..<count).map { outputIndex in
        var sum = Complex<Double>.zero
        for inputIndex in 0..<count {
            let angle = sign * 2.0 * Double.pi * Double(inputIndex * outputIndex) / Double(count)
            let twiddle = Complex<Double>(length: 1.0, phase: angle)
            sum += input[inputIndex] * twiddle
        }
        return Complex(sum.real * scale, sum.imaginary * scale)
    }
}

func expectedForwardAxis(count: Int, spacing: Double) -> [Double] {
    guard count > 0 else { return [] }

    let positiveCount = (count + 1) / 2
    let factor = 2.0 * Double.pi / (Double(count) * spacing)
    return (0..<count).map { index in
        let signedIndex = index < positiveCount ? index : index - count
        return Double(signedIndex) * factor
    }
}

func expectedInverseAxis(count: Int, spacing: Double) -> [Double] {
    guard count > 0 else { return [] }

    let factor = 2.0 * Double.pi / (Double(count) * spacing)
    return (0..<count).map { Double($0) * factor }
}

func expectForwardPlanMatchesOracle(
    _ plan: any FFT.Plan,
    input: [Complex<Double>],
    spacing: Double
) {
    let (axis, output) = plan.execute(input, spacing: spacing)
    expectApproximatelyEqual(axis, expectedForwardAxis(count: input.count, spacing: spacing))
    expectApproximatelyEqual(output, naiveDFT(input))
}

func expectInversePlanMatchesOracle(
    _ plan: any FFT.Plan,
    input: [Complex<Double>],
    spacing: Double
) {
    let (axis, output) = plan.execute(input, spacing: spacing)
    expectApproximatelyEqual(axis, expectedInverseAxis(count: input.count, spacing: spacing))
    expectApproximatelyEqual(output, naiveDFT(input, inverse: true))
}
