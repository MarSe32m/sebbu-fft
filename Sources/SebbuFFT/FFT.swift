//
// FFT.swift
// sebbu-fft
//
// Created by Sebastian Toivonen on 4.8.2026
//

import RealModule
import ComplexModule

/// Discrete Fourier transform operations and reusable transform plans.
///
/// `FFT` provides one-shot forward and inverse transforms for arrays of
/// `Complex<Double>`, in-place helpers for changing bin order, and the ``FFT/Plan``
/// protocol used by alternative backends.
///
/// The forward transform is unnormalized and uses a negative exponent. The inverse
/// transform uses a positive exponent and applies the conventional `1 / N` scale.
/// Frequency axes are expressed as angular frequency in radians per input-axis unit.
public enum FFT {
    /// Returns whether a sample count is preferred by the default backend.
    ///
    /// On macOS, compatible counts are at least eight and have the form
    /// `f * 2^k`, where `f` is `1`, `3`, `5`, or `15`. On other platforms, every
    /// count of at least eight is considered compatible with the portable backend.
    ///
    /// This query describes the package's default backend selection. An explicitly
    /// chosen backend may support a different set of sizes.
    ///
    /// - Parameter n: The proposed number of complex samples.
    /// - Returns: `true` when `n` is considered compatible on the current platform.
    @inlinable
    public static func isCompatibleSampleSize(_ n: Int) -> Bool {
        #if os(macOS)
        guard n >= 8 else { return false }

        // Remove the 2^k component. The remaining odd factor must be allowed.
        var oddFactor = n
        while oddFactor.isMultiple(of: 2) {
            oddFactor /= 2
        }

        switch oddFactor {
        case 1, 3, 5, 15:
            return true
        default:
            return false
        }
        #else
        return n >= 8
        #endif
    }

    /// Returns the smallest compatible sample count that is not less than a request.
    ///
    /// Use this function when padding a signal for the package's default backend.
    ///
    /// - Parameter n: The minimum acceptable sample count.
    /// - Returns: The smallest compatible count greater than or equal to `n`.
    /// - Precondition: `n >= 8` and the result is representable as an `Int`.
    @inlinable
    public static func nextCompatibleSampleSize(_ n: Int) -> Int {
        if n < 8 { return 8 }
        
        #if os(macOS)
        let factors: [_ of Int] = [1, 3, 5, 15]
        var bestCandidate: Int?
        for i in factors.indices {
            var candidate = factors[i]
            while candidate < n {
                let (doubled, overflow) = candidate.multipliedReportingOverflow(by: 2)
                if overflow { break }
                candidate = doubled
            }
            guard candidate >= n else { continue }
            if let currentBest = bestCandidate {
                if candidate < currentBest {
                    bestCandidate = candidate
                }
            } else {
                bestCandidate = candidate
            }
        }
        guard let bestCandidate else {
            preconditionFailure(
                "No compatible sample count is representable as an Int."
            )
        }

        return bestCandidate
        #else
        return n
        #endif
    }

    /// Computes the discrete Fourier transform of equally spaced complex samples.
    ///
    /// For `N` samples, the transform follows this convention:
    ///
    /// ```text
    /// X[k] = sum(x[n] * exp(-i * 2π * n * k / N), n = 0 ..< N)
    /// ```
    ///
    /// The result is unnormalized. Frequencies are returned in unshifted order, with
    /// zero and positive frequencies followed by negative frequencies. Use
    /// ``fftShift(frequencies:spectrum:)`` when a centered frequency axis is more useful.
    ///
    /// - Parameters:
    ///   - input: The equally spaced complex input samples.
    ///   - spacing: The distance between adjacent samples. Use a finite, positive
    ///     value. A spacing measured in seconds produces angular frequencies in
    ///     radians per second.
    /// - Returns: The angular-frequency bins and corresponding complex spectrum. Each
    ///   output array has `ipnut.count` elements; empty input produces two empty arrays.
    @inlinable
    nonisolated public static func fft(_ input: [Complex<Double>], spacing: Double = 1.0) -> (frequencies: [Double], spectrum: [Complex<Double>]) {
        if input.isEmpty { return ([], []) }
        let plan = defaultFFTPlan(sampleSize: input.count)
        let (frequencies, spectrum) = plan.execute(input, spacing: spacing)
        return (frequencies, spectrum)
    }

    /// Computes the inverse discrete Fourier transform of complex frequency bins.
    ///
    /// For `N` bins, the transform follows this convention:
    ///
    /// ```text
    /// x[n] = (1 / N) * sum(X[k] * exp(i * 2π * n * k / N), k = 0 ..< N)
    /// ```
    ///
    /// Supply bins in the unshifted order produced by ``fft(_:spacing:)``. If the bins
    /// are centered, call ``ifftShift(_:_:)`` first.
    ///
    /// - Parameters:
    ///   - input: The complex frequency bins in unshifted order.
    ///   - spacing: The angular-frequency distance between adjacent bins. Use a finite,
    ///     positive value. The returned sample positions are separated by
    ///     `2π / (N * spacing)`.
    /// - Returns: The reconstructed sample positions and normalized complex signal.
    ///   Each output array has `input.count` elements; empty input produces two empty arrays.
    @inlinable
    nonisolated public static func ifft(_ input: [Complex<Double>], spacing: Double = 1.0) -> (t: [Double], signal: [Complex<Double>]) {
        if input.isEmpty { return ([], []) }
        let plan = defaultiFFTPlan(sampleSize: input.count)
        let (t, signal) = plan.execute(input, spacing: spacing)
        return (t, signal)
    }

    /// Moves the zero-frequency bin to the center of two arrays in place.
    ///
    /// The frequency and spectrum arrays are reordered in lockstep so that every
    /// spectrum value remains paired with its frequency. Applying ``ifftShift(_:_:)``
    /// restores the original unshifted order. Empty and single-element inputs are
    /// left unchanged.
    ///
    /// - Parameters:
    ///   - frequencies: The angular-frequency axis to reorder.
    ///   - spectrum: The spectrum values to reorder in the same way.
    /// - Precondition: Both arrays have the same number of elements.
    @inlinable
    nonisolated public static func fftShift(frequencies: inout [Double], spectrum: inout [Complex<Double>]) {
        var frequencySpan = frequencies.mutableSpan
        var spectrumSpan = spectrum.mutableSpan
        fftShift(&frequencySpan, &spectrumSpan)
    }

    /// Moves the zero-frequency bin to the center of two mutable spans in place.
    ///
    /// This overload performs the same operation as
    /// ``fftShift(frequencies:spectrum:)`` without requiring array ownership.
    ///
    /// - Parameters:
    ///   - frequencies: A mutable span containing the angular-frequency axis.
    ///   - spectrum: A mutable span containing the paired spectrum values.
    /// - Precondition: Both spans have the same number of elements.
    @inlinable
    nonisolated public static func fftShift(_ frequencies: inout MutableSpan<Double>, _ spectrum: inout MutableSpan<Complex<Double>>) {
        precondition(frequencies.count == spectrum.count, "The frequencies and spectrum must match in size.")
        let N = spectrum.count
        guard N > 1 else { return }
        if N.isMultiple(of: 2) {
            for i in 0..<N/2 {
                spectrum.swapAt(i, i + N/2)
                frequencies.swapAt(i, i + N/2)
            }
        } else {
            // Swap non-zero frequencies
            for i in 1...N/2 {
                // Swap non-zero frequencies
                spectrum.swapAt(i, i + N/2)
                frequencies.swapAt(i, i + N/2)
            }
            // Shift the zero to the middle
            for i in 0..<N/2 {
                spectrum.swapAt(i, i + 1)
                frequencies.swapAt(i, i + 1)
            }
        }
    }

    /// Restores two centered arrays to unshifted FFT bin order in place.
    ///
    /// This function is the inverse permutation of
    /// ``fftShift(frequencies:spectrum:)``. It moves the zero-frequency bin back to
    /// index zero while preserving frequency–spectrum pairs.
    ///
    /// - Parameters:
    ///   - frequencies: The centered angular-frequency axis to reorder.
    ///   - spectrum: The centered spectrum values to reorder in the same way.
    /// - Precondition: Both arrays have the same number of elements.
    @inlinable
    nonisolated public static func ifftShift(_ frequencies: inout [Double], _ spectrum: inout [Complex<Double>]) {
        var frequencySpan = frequencies.mutableSpan
        var spectrumSpan = spectrum.mutableSpan
        ifftShift(&frequencySpan, &spectrumSpan)
    }

    /// Restores two centered mutable spans to unshifted FFT bin order in place.
    ///
    /// This overload performs the same operation as the array-based
    /// ``ifftShift(_:_:)`` without requiring array ownership.
    ///
    /// - Parameters:
    ///   - frequencies: A mutable span containing the centered angular-frequency axis.
    ///   - spectrum: A mutable span containing the paired centered spectrum values.
    /// - Precondition: Both spans have the same number of elements.
    @inlinable
    nonisolated public static func ifftShift(_ frequencies: inout MutableSpan<Double>, _ spectrum: inout MutableSpan<Complex<Double>>) {
        precondition(frequencies.count == spectrum.count, "The frequencies and spectrum must match in size.")
        let N = spectrum.count
        if N.isMultiple(of: 2) {
            // Same as fftShift
            fftShift(&frequencies, &spectrum)
        } else {
            // Swap non-zero elements
            for i in 0..<N/2 {
                spectrum.swapAt(i, i + N/2 + 1)
                frequencies.swapAt(i, i + N/2 + 1)
            }
            // Shift the zero to the left
            for i in stride(from: N/2, to: 0, by: -1) {
                spectrum.swapAt(i, i - 1)
                frequencies.swapAt(i, i - 1)
            }
        }
    }
}