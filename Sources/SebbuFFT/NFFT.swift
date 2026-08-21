// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import RealModule

public extension FFT {
    /// Evaluates centered Fourier coefficients at non-equispaced positions
    /// using the direct Fourier sum.
    ///
    /// For an even number `N` of coefficients, coefficient index `r`
    /// represents the wave number `k = r - N / 2`, and this function computes
    ///
    /// ```text
    /// f[j] = sum(fHat[r] * exp(-2πi * k * positions[j]), r = 0 ..< N).
    /// ```
    ///
    /// Positions are measured in turns and are periodic with period one.
    /// This direct `O(MN)` implementation is useful for small problems and as
    /// an accuracy oracle for ``nfft(_:at:configuration:kernel:)``.
    ///
    /// - Parameters:
    ///   - coefficients: A positive, even number of Fourier coefficients in
    ///     centered order, from wave number `-N / 2` through `N / 2 - 1`.
    ///   - positions: Finite evaluation positions. Values outside
    ///     `[-0.5, 0.5)` are wrapped periodically.
    /// - Returns: The Fourier sum evaluated at each supplied position.
    @inlinable
    nonisolated static func ndft(
        _ coefficients: [Complex<Double>],
        at positions: [Double]
    ) -> [Complex<Double>] {
        preconditionValidNFFTCoefficientCount(coefficients.count)
        preconditionFiniteNFFTPositions(positions)

        let halfCount = coefficients.count / 2
        return positions.map { position in
            let wrappedPosition = nfftWrappedPosition(position)
            var value = Complex<Double>.zero
            for coefficientIndex in coefficients.indices {
                let waveNumber = coefficientIndex - halfCount
                let phase = -2.0 * Double.pi * Double(waveNumber)
                    * wrappedPosition
                value += coefficients[coefficientIndex]
                    * Complex<Double>(length: 1.0, phase: phase)
            }
            return value
        }
    }

    /// Applies the adjoint direct non-equispaced Fourier transform.
    ///
    /// For centered wave numbers `k = -N / 2, ..., N / 2 - 1`, this computes
    ///
    /// ```text
    /// fHat[k] = sum(values[j] * exp(2πi * k * positions[j]), j = 0 ..< M).
    /// ```
    ///
    /// - Parameters:
    ///   - values: Complex values at the non-equispaced positions.
    ///   - positions: One finite position for every value. Positions are
    ///     measured in turns and wrapped with period one.
    ///   - coefficientCount: The positive, even number of centered Fourier
    ///     coefficients to return.
    /// - Returns: The adjoint Fourier sum in centered wave-number order.
    @inlinable
    nonisolated static func ndftAdjoint(
        _ values: [Complex<Double>],
        at positions: [Double],
        coefficientCount: Int
    ) -> [Complex<Double>] {
        precondition(values.count == positions.count, "The NFFT positions and values must have the same count.")
        preconditionValidNFFTCoefficientCount(coefficientCount)
        preconditionFiniteNFFTPositions(positions)

        let halfCount = coefficientCount / 2
        return (0..<coefficientCount).map { coefficientIndex in
            let waveNumber = coefficientIndex - halfCount
            var coefficient = Complex<Double>.zero
            for sampleIndex in positions.indices {
                let phase = 2.0 * Double.pi * Double(waveNumber)
                    * nfftWrappedPosition(positions[sampleIndex])
                coefficient += values[sampleIndex]
                    * Complex<Double>(length: 1.0, phase: phase)
            }
            return coefficient
        }
    }

    /// Evaluates centered Fourier coefficients at non-equispaced positions
    /// using an oversampled fast Fourier transform.
    ///
    /// Create an ``NFFTPlan`` directly when repeatedly transforming coefficient
    /// arrays with the same size and positions; the plan caches interpolation
    /// weights and its underlying FFT plan.
    ///
    /// - Parameters:
    ///   - coefficients: A positive, even number of Fourier coefficients in
    ///     centered wave-number order.
    ///   - positions: Finite evaluation positions measured in turns.
    ///   - configuration: Oversampling and accuracy controls.
    ///   - kernel: The interpolation kernel. The Gaussian kernel is the default.
    /// - Returns: The approximate Fourier sum at each position.
    @inlinable
    nonisolated static func nfft(
        _ coefficients: [Complex<Double>],
        at positions: [Double],
        configuration: NFFTConfiguration = .init(),
        kernel: any NFFTKernel = GaussianNFFTKernel()
    ) -> [Complex<Double>] {
        let plan = NFFTPlan(
            positions: positions,
            coefficientCount: coefficients.count,
            configuration: configuration,
            kernel: kernel
        )
        return plan.execute(coefficients)
    }

    /// Applies the adjoint non-equispaced fast Fourier transform.
    ///
    /// Create an ``NFFTAdjointPlan`` directly when repeatedly transforming
    /// values at the same positions.
    ///
    /// - Parameters:
    ///   - values: Complex values at the non-equispaced positions.
    ///   - positions: One finite position for every value, measured in turns.
    ///   - coefficientCount: The positive, even number of centered Fourier
    ///     coefficients to return.
    ///   - configuration: Oversampling and accuracy controls.
    ///   - kernel: The interpolation kernel. The Gaussian kernel is the default.
    /// - Returns: The approximate adjoint transform in centered order.
    @inlinable
    nonisolated static func nfftAdjoint(
        _ values: [Complex<Double>],
        at positions: [Double],
        coefficientCount: Int,
        configuration: NFFTConfiguration = .init(),
        kernel: any NFFTKernel = GaussianNFFTKernel()
    ) -> [Complex<Double>] {
        precondition(values.count == positions.count, "The NFFT positions and values must have the same count.")
        let plan = NFFTAdjointPlan(
            positions: positions,
            coefficientCount: coefficientCount,
            configuration: configuration,
            kernel: kernel
        )
        return plan.execute(values)
    }
}

internal extension FFT {
    @inlinable
    static func preconditionValidNFFTCoefficientCount(_ count: Int) {
        precondition(
            count > 0 && count.isMultiple(of: 2),
            "An NFFT requires a positive, even number of Fourier coefficients."
        )
    }
    
    @inlinable
    static func preconditionFiniteNFFTPositions(_ positions: [Double]) {
        precondition(
            positions.allSatisfy(\.isFinite),
            "NFFT positions must be finite."
        )
    }

    /// Reduces a position to the half-open periodic interval `[-0.5, 0.5)`.
    @inlinable
    static func nfftWrappedPosition(_ position: Double) -> Double {
        let translated = position + 0.5
        return translated - translated.rounded(.down) - 0.5
    }
    
    @inlinable
    static func nfftPositiveModulo(_ value: Int, modulus: Int) -> Int {
        let remainder = value % modulus
        return remainder >= 0 ? remainder : remainder + modulus
    }
    
    @inlinable
    @inline(always)
    static func nfftScale(
        _ value: Complex<Double>,
        by scale: Double
    ) -> Complex<Double> {
        Complex(value.real * scale, value.imaginary * scale)
    }
}
