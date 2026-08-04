// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import Numerics

public extension FFT {
    /// A reusable strategy for computing a discrete Fourier transform.
    ///
    /// A plan encapsulates a transform direction and, optionally, backend-specific
    /// resources such as precomputed tables or native-library handles. The package's
    /// built-in forward and inverse plans both conform to this protocol, and companion
    /// packages can provide additional implementations without adding those backends as
    /// dependencies of `sebbu-fft`.
    ///
    /// The meaning of `spacing` and the returned axis is direction-specific:
    ///
    /// - A forward plan receives the spacing between input samples and returns angular
    ///   frequencies in radians per input-axis unit.
    /// - An inverse plan receives the spacing between angular-frequency bins and returns
    ///   sample positions in the reciprocal unit.
    ///
    /// Implementations should preserve the transform conventions used by ``FFT``: an
    /// unnormalized forward transform, an inverse transform normalized by `1 / N`, and
    /// unshifted bin order unless documented otherwise.
    protocol Plan {
        /// Executes the transform represented by this plan.
        ///
        /// - Parameters:
        ///   - input: The complex input values. For fixed-size plans, `input.count` must equal
        ///     the sample size supplied when the plan was created.
        ///   - spacing: The distance between adjacent input-axis values. Use a finite,
        ///     positive value. Its units and the returned axis depend on the plan's
        ///     transform direction.
        /// - Returns: A pair containing the output axis in `x` and the transformed
        ///   complex values in `y`. Both arrays have the same count as the input.
        func execute(_ input: [Complex<Double>], spacing: Double) -> (x: [Double], y: [Complex<Double>])
    }

    /// Creates the plan used by ``FFT/fft(_:spacing:)`` for the current platform.
    ///
    /// Keep the returned plan and execute it repeatedly when processing multiple blocks
    /// with the same size. The concrete backend is an implementation detail and can vary
    /// by platform.
    ///
    /// - Parameter sampleSize: The number of complex values in each input block.
    /// - Returns: The package's default forward-transform plan.
    static func defaultFFTPlan(sampleSize: Int) -> some Plan {
        #if os(macOS)
        return AccelerateFFTPlan(sampleSize: sampleSize)
        #else
        return FFTPlan(sampleSize: sampleSize)
        #endif
    }

    /// Creates the plan used by ``FFT/ifft(_:spacing:)`` for the current platform.
    ///
    /// Keep the returned plan and execute it repeatedly when processing multiple blocks
    /// with the same size. The concrete backend is an implementation detail and can vary
    /// by platform.
    ///
    /// - Parameter sampleSize: The number of complex values in each input block.
    /// - Returns: The package's default inverse-transform plan.
    static func defaultiFFTPlan(sampleSize: Int) -> some Plan {
        #if os(macOS)
        return AccelerateiFFTPlan(sampleSize: sampleSize)
        #else
        return iFFTPlan(sampleSize: sampleSize)
        #endif
    }
}
