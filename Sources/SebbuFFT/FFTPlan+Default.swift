//
// FFTPlan+Default.swift
// sebbu-fft
//
// Created by Sebastian Toivonen on 4.8.2026
//

import Numerics

public extension FFT {
    /// A portable, pure-Swift forward-transform plan.
    ///
    /// Power-of-two inputs use a radix-2 Cooley–Tukey transform. Other nonempty
    /// sizes use Bluestein's algorithm, allowing the plan to transform arbitrary
    /// sample counts without a native FFT dependency.
    struct FFTPlan: Sendable, Plan {
        /// The input size this plan was created to transform.
        ///
        /// Pass arrays of this size to ``execute(_:spacing:)``. The pure-Swift
        /// implementation derives its work size from the array, but matching the
        /// declared size keeps code portable to fixed-size backends.
        public let sampleSize: Int

        /// Creates a pure-Swift forward-transform plan.
        ///
        /// - Parameter sampleSize: The number of values in each input block.
        @inlinable
        public init(sampleSize: Int) {
            self.sampleSize = sampleSize
        }

        /// Computes an unnormalized forward discrete Fourier transform.
        ///
        /// The spectrum is returned in unshifted bin order: zero and positive
        /// frequencies first, followed by negative frequencies. Use
        /// ``FFT/fftShift(frequencies:spectrum:)`` to center the zero-frequency bin.
        ///
        /// - Parameters:
        ///   - input: Equally spaced complex samples.
        ///   - spacing: The distance between adjacent samples. Use a finite, positive
        ///     value.
        /// - Returns: Angular frequencies in radians per input-axis unit and the complex
        ///   spectrum at those frequencies. Empty input produces two empty arrays.
        @inlinable
        public func execute(_ input: [Complex<Double>], spacing: Double) -> (x: [Double], y: [Complex<Double>]) {
            let N = input.count
            guard N > 0 else { return ([], []) }

            let spectrum: [Complex<Double>]
            if N & (N - 1) == 0 {
                spectrum = FFT.radix2FFT(input)
            } else {
                spectrum = FFT.bluesteinFFT(input)
            }
            let frequencies: [Double] = [Double](unsafeUninitializedCapacity: N) { buffer, initializedCount in
                initializedCount = N
                let upperBound = N.isMultiple(of: 2) ? N/2 : (N/2 + 1)
                let lowerRange = (0..<upperBound)
                let upperRange = (-N/2..<0)
                let denominator = spacing * Double(N)
                let factor = 2.0 * .pi / denominator
                var index = 0
                for i in lowerRange {
                    buffer.initializeElement(at: index, to: Double(i) * factor)
                    index += 1
                }
                for i in upperRange {
                    buffer.initializeElement(at: index, to: Double(i) * factor)
                    index += 1
                }
            }
            return (frequencies, spectrum)
        }
    }

    /// A portable, pure-Swift inverse-transform plan.
    ///
    /// Power-of-two inputs use a radix-2 Cooley–Tukey transform. Other nonempty
    /// sizes use Bluestein's algorithm. The result is normalized by `1 / N`, making
    /// it the inverse of the package's unnormalized forward transform.
    struct iFFTPlan: Sendable, Plan {
        /// The input size this plan was created to transform.
        ///
        /// Pass arrays of this size to ``execute(_:spacing:)``. The pure-Swift
        /// implementation derives its work size from the array, but matching the
        /// declared size keeps code portable to fixed-size backends.
        public let sampleSize: Int

        /// Creates a pure-Swift inverse-transform plan.
        ///
        /// - Parameter sampleSize: The number of frequency bins in each input block.
        @inlinable
        public init(sampleSize: Int) {
            self.sampleSize = sampleSize
        }

        /// Computes a normalized inverse discrete Fourier transform.
        ///
        /// - Parameters:
        ///   - input: Complex frequency bins in unshifted order.
        ///   - spacing: The angular-frequency distance between adjacent bins. Use a
        ///     finite, positive value.
        /// - Returns: Sample positions separated by `2π / (N * spacing)` and the
        ///   reconstructed signal, normalized by `1 / N`. Empty input produces two
        ///   empty arrays.
        @inlinable
        public func execute(_ input: [Complex<Double>], spacing: Double) -> (x: [Double], y: [Complex<Double>]) {
            let N = input.count
            guard N > 0 else { return ([], []) }
            let conjugate = input.map { $0.conjugate }
            var signal: [Complex<Double>]
            if N & (N - 1) == 0 {
                signal = FFT.radix2FFT(conjugate).map { Complex($0.real / Double(N), -$0.imaginary / Double(N)) }
            } else {
                signal = FFT.bluesteinFFT(conjugate).map { Complex($0.real / Double(N), -$0.imaginary / Double(N)) }
            }
            let range = (0..<N)
            let factor = 2.0 * .pi / (spacing * Double(N))
            let t = range.map { Double($0) * factor }
            return (t, signal)
        }
    }
}

internal extension FFT {
    /// Computes an arbitrary-length forward DFT with Bluestein's chirp-z algorithm.
    ///
    /// The transform is unnormalized and follows the same negative-exponent convention
    /// as ``radix2FFT(_:)``.
    ///
    /// - Parameter x: A nonempty array of complex samples.
    /// - Returns: The unnormalized forward transform of `x`.
    @inlinable
    static func bluesteinFFT(_ input: [Complex<Double>]) -> [Complex<Double>] {
        let N = input.count
        let m = (2 * N - 1).nextPowerOfTwo
        let piOverN = Double.pi / Double(N)
        var a = [Complex<Double>](repeating: .zero, count: m)
        var b = [Complex<Double>](repeating: .zero, count: m)
        for n in 0..<N {
            let angle = piOverN * Double(n * n)
            let w = Complex(length: 1.0, phase: -angle)
            a[n] = Relaxed.product(input[n], w)
            b[n] = w.conjugate
            if n != 0 {
                b[m - n] = b[n]
            }
        }

        // Perform convolution via FFT
        let A = radix2FFT(a)
        let B = radix2FFT(b)

        var C = [Complex<Double>](repeating: .zero, count: m)
        for i in 0..<m {
            C[i] = Relaxed.product(A[i], B[i])
        }
        let scalingFactor = 1 / Double(m)
        let c = radix2iFFT(C).map { Complex(Relaxed.product($0.real, scalingFactor),
                                               Relaxed.product($0.imaginary, scalingFactor)) }

        var result = [Complex<Double>](repeating: .zero, count: N)
        for k in 0..<N {
            let angle = piOverN * Double(k * k)
            let w = Complex(length: 1.0, phase: -angle)
            result[k] = Relaxed.product(c[k], w)
        }

        return result
    }

    /// Computes an unnormalized radix-2 Cooley–Tukey forward transform.
    ///
    /// - Parameter input: A nonempty array whose count is a power of two.
    /// - Returns: The unnormalized forward transform in unshifted bin order.
    /// - Precondition: `input.count` is a nonzero power of two.
    @inlinable
    static func radix2FFT(_ input: [Complex<Double>]) -> [Complex<Double>] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Input size must be a power of 2")

        var a = input

        let logN = n.log2
        for i in 0..<n {
            let j = bitReversed(i, bits: logN)
            if i < j {
                a.swapAt(i, j)
            }
        }

        var m = 2
        while m <= n {
            let theta = -2.0 * Double.pi / Double(m)
            let wm = Complex<Double>(length: 1.0, phase: theta)
            for k in stride(from: 0, to: n, by: m) {
                var w = Complex<Double>(1, 0)
                for j in 0..<(m / 2) {
                    let t = Relaxed.product(w, a[k + j + m / 2])
                    let u = a[k + j]
                    a[k + j] = Relaxed.sum(u, t)
                    a[k + j + m / 2] = Relaxed.sum(u, -t)
                    w = Relaxed.product(w, wm)
                }
            }
            m *= 2
        }

        return a
    }

    /// Computes an unnormalized radix-2 inverse transform.
    ///
    /// This helper applies conjugation around ``radix2FFT(_:)`` but deliberately does
    /// not apply the `1 / N` scale factor. Callers that need the conventional inverse
    /// normalization must apply it separately.
    ///
    /// - Parameter input: A nonempty array whose count is a power of two.
    /// - Returns: The unnormalized inverse transform in unshifted order.
    @inlinable
    static func radix2iFFT(_ input: [Complex<Double>]) -> [Complex<Double>] {
        let conjugate = input.map { $0.conjugate }
        let result = radix2FFT(conjugate)
        return result.map { $0.conjugate }
    }

    /// Reverses the lowest `bits` bits in an integer index.
    ///
    /// - Parameters:
    ///   - x: The nonnegative index to reverse.
    ///   - bits: The number of low-order bits to include.
    /// - Returns: The bit-reversed index used by the radix-2 permutation.
    @inline(always)
    @inlinable
    static func bitReversed(_ x: Int, bits: Int) -> Int {
        var x = x
        var result = 0
        for _ in 0..<bits {
            result = (result << 1) | (x & 1)
            x >>= 1
        }
        return result
    }
}