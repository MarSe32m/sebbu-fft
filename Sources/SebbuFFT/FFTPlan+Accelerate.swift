// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

#if canImport(Accelerate)
import Numerics
import Accelerate

public extension FFT {
    /// A forward-transform plan backed by Accelerate's double-precision DFT routines.
    ///
    /// The plan owns an Accelerate setup and can reuse it across calls. If Accelerate
    /// cannot create that setup, ``execute(_:spacing:)`` falls back to ``FFTPlan``.
    /// Instances are available only on platforms that can import Accelerate.
    final class AccelerateFFTPlan: Plan {
        /// The Accelerate setup owned by this plan, or `nil` when setup creation failed.
        internal let setup: OpaquePointer?

        /// Creates a forward plan for a fixed number of complex samples.
        ///
        /// - Parameter sampleSize: The number of values passed to every subsequent call
        ///   to ``execute(_:spacing:)``. Use a size supported by Accelerate; see
        ///   ``FFT/isCompatibleSampleSize(_:)``.
        public init(sampleSize: Int) {
            self.setup = vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(sampleSize), .FORWARD)
        }

        /// Computes an unnormalized forward discrete Fourier transform.
        ///
        /// The returned angular-frequency bins use unshifted order: zero and positive
        /// frequencies first, followed by negative frequencies.
        ///
        /// - Parameters:
        ///   - input: Equally spaced complex samples. Its count must equal the size used to
        ///     initialize this plan.
        ///   - spacing: The distance between adjacent samples. Use a finite, positive
        ///     value.
        /// - Returns: Angular frequencies in radians per input-axis unit and the complex
        ///   spectrum at those frequencies.
        public func execute(_ input: [Complex<Double>], spacing: Double) -> (x: [Double], y: [Complex<Double>]) {
            let N = input.count
            let frequencies: [Double] = [Double](unsafeUninitializedCapacity: N) { buffer, initializedCount in
                initializedCount = N
                let upperBound = N.isMultiple(of: 2) ? N/2 : (N/2 + 1)
                let lowerRange = (0..<upperBound)
                let upperRange = (-N/2..<0)
                let denominator = spacing * Double(N)
                var index = 0
                for i in lowerRange {
                    buffer.initializeElement(at: index, to: 2.0 * .pi * Double(i) / denominator)
                    index += 1
                }
                for i in upperRange {
                    buffer.initializeElement(at: index, to: 2.0 * .pi * Double(i) / denominator)
                    index += 1
                }
            }

            if let setup = setup {
                let realParts = input.map { $0.real }
                let imaginaryParts = input.map { $0.imaginary }
                var realPartsOutput = [Double](repeating: 0, count: N)
                var imaginaryPartsOutput = [Double](repeating: 0, count: N)
                vDSP_DFT_ExecuteD(setup, realParts, imaginaryParts, &realPartsOutput, &imaginaryPartsOutput)
                let spectrum = zip(realPartsOutput, imaginaryPartsOutput).map { Complex($0, $1) }
                return (frequencies, spectrum)
            }
            // Fall back to default implementation
            let plan = FFTPlan(sampleSize: N)
            return plan.execute(input, spacing: spacing)
        }

        deinit {
            if let setup {
                vDSP_DFT_DestroySetupD(setup)
            }
        }
    }

    /// An inverse-transform plan backed by Accelerate's double-precision DFT routines.
    ///
    /// The plan owns an Accelerate setup and can reuse it across calls. If Accelerate
    /// cannot create that setup, ``execute(_:spacing:)`` falls back to ``iFFTPlan``.
    /// Instances are available only on platforms that can import Accelerate.
    final class AccelerateiFFTPlan: Plan {
        /// The Accelerate setup owned by this plan, or `nil` when setup creation failed.
        internal let setup: OpaquePointer?

        /// Creates an inverse plan for a fixed number of complex frequency bins.
        ///
        /// - Parameter sampleSize: The number of values passed to every subsequent call
        ///   to ``execute(_:spacing:)``. Use a size supported by Accelerate; see
        ///   ``FFT/isCompatibleSampleSize(_:)``.
        public init(sampleSize: Int) {
            self.setup = vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(sampleSize), .INVERSE)
        }

        /// Computes a normalized inverse discrete Fourier transform.
        ///
        /// - Parameters:
        ///   - input: Complex frequency bins in unshifted order. Its count must equal the
        ///     size used to initialize this plan.
        ///   - spacing: The angular-frequency distance between adjacent bins. Use a
        ///     finite, positive value.
        /// - Returns: Sample positions separated by `2π / (N * spacing)` and the
        ///   reconstructed signal, normalized by `1 / N`.
        public func execute(_ input: [Complex<Double>], spacing: Double) -> (x: [Double], y: [Complex<Double>]) {
            let N = input.count
            let range = (0..<N)
            let denominator = spacing * Double(N)
            let t = range.map { 2.0 * .pi * Double($0) / denominator }

            if let setup = setup {
                let realParts = input.map { $0.real }
                let imaginaryParts = input.map { $0.imaginary }
                var realPartsOutput = [Double](repeating: 0, count: N)
                var imaginaryPartsOutput = [Double](repeating: 0, count: N)
                vDSP_DFT_ExecuteD(setup, realParts, imaginaryParts, &realPartsOutput, &imaginaryPartsOutput)
                let signal = zip(realPartsOutput, imaginaryPartsOutput).map { Complex($0 / Double(N), $1 / Double(N)) }
                return (t, signal)
            }
            // Fall back to default implementation
            let plan = iFFTPlan(sampleSize: N)
            return plan.execute(input, spacing: spacing)
        }

        deinit {
            if let setup {
                vDSP_DFT_DestroySetupD(setup)
            }
        }
    }
}
#endif
