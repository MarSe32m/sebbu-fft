// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import RealModule

public extension FFT {
    /// A window used to interpolate between an oversampled FFT grid and
    /// non-equispaced positions.
    ///
    /// Kernel values are real. This makes the interpolation used by
    /// ``NFFTPlan`` and the transpose interpolation used by
    /// ``NFFTAdjointPlan`` an adjoint pair.
    protocol NFFTKernel: Sendable {
        /// Evaluates the spatial interpolation window.
        ///
        /// - Parameters:
        ///   - distance: The periodic distance from a non-equispaced position
        ///     to a regular-grid position, expressed in turns.
        ///   - gridSize: The number of points in the oversampled FFT grid.
        ///   - windowHalfWidth: Half the number of grid points retained by the
        ///     truncated interpolation window.
        ///   - oversamplingFactor: The ratio between `gridSize` and the number
        ///     of Fourier coefficients.
        func value(
            at distance: Double,
            gridSize: Int,
            windowHalfWidth: Int,
            oversamplingFactor: Int
        ) -> Double

        /// Evaluates the Fourier transform of the interpolation window.
        ///
        /// The returned value uses the same normalization as the discrete
        /// Fourier coefficients of a window sampled on `gridSize` points.
        func fourierValue(
            at waveNumber: Int,
            gridSize: Int,
            windowHalfWidth: Int,
            oversamplingFactor: Int
        ) -> Double

        /// Estimates a window half-width for a requested approximation
        /// tolerance.
        func estimateWindowHalfWidth(
            tolerance: Double,
            coefficientCount: Int,
            oversamplingFactor: Int
        ) -> Int
    }

    /// The Gaussian interpolation window used by the default NFFT plans.
    ///
    /// This window follows the one-dimensional Gaussian construction used by
    /// Jake VanderPlas's `nfft` package. It is inexpensive to evaluate and its
    /// Fourier transform is available analytically.
    struct GaussianNFFTKernel: NFFTKernel {
        /// Creates a Gaussian NFFT interpolation kernel.
        public init() {}
        
        @inlinable
        public func value(
            at distance: Double,
            gridSize: Int,
            windowHalfWidth: Int,
            oversamplingFactor: Int
        ) -> Double {
            let b = shapeParameter(
                windowHalfWidth: windowHalfWidth,
                oversamplingFactor: oversamplingFactor
            )
            let scaledDistance = Double(gridSize) * distance
            return Double.exp(
                -(scaledDistance * scaledDistance) / b
            ) / (Double.pi * b).squareRoot()
        }
        
        @inlinable
        public func fourierValue(
            at waveNumber: Int,
            gridSize: Int,
            windowHalfWidth: Int,
            oversamplingFactor: Int
        ) -> Double {
            let b = shapeParameter(
                windowHalfWidth: windowHalfWidth,
                oversamplingFactor: oversamplingFactor
            )
            let argument = Double.pi * Double(waveNumber) / Double(gridSize)
            return Double.exp(-b * argument * argument) / Double(gridSize)
        }
        
        @inlinable
        public func estimateWindowHalfWidth(
            tolerance: Double,
            coefficientCount: Int,
            oversamplingFactor: Int
        ) -> Int {
            let estimatedErrorConstant = tolerance / Double(coefficientCount)
            let sigma = Double(oversamplingFactor)
            let denominator = Double.pi * (1.0 - 1.0 / (2.0 * sigma - 1.0))
            let estimate = -Double.log(0.25 * estimatedErrorConstant)
                / denominator
            return Int(estimate.rounded(.up))
        }

        /// Returns the analytic error constant associated with a window
        /// half-width and oversampling factor.
        @inlinable
        public func errorConstant(
            windowHalfWidth: Int,
            oversamplingFactor: Int
        ) -> Double {
            let sigma = Double(oversamplingFactor)
            let exponent = -Double(windowHalfWidth) * Double.pi
                * (1.0 - 1.0 / (2.0 * sigma - 1.0))
            return 4.0 * Double.exp(exponent)
        }
        
        @inlinable
        internal func shapeParameter(
            windowHalfWidth: Int,
            oversamplingFactor: Int
        ) -> Double {
            let sigma = Double(oversamplingFactor)
            return 2.0 * sigma * Double(windowHalfWidth)
                / ((2.0 * sigma - 1.0) * Double.pi)
        }
    }

    /// Accuracy and oversampling controls shared by forward and adjoint NFFTs.
    struct NFFTConfiguration: Sendable {
        /// The size multiplier used to construct the regular FFT grid.
        public let oversamplingFactor: Int

        /// The requested truncation-tolerance parameter used to estimate the
        /// interpolation window when ``windowHalfWidth`` is `nil`.
        public let tolerance: Double

        /// An explicit half-width for the interpolation window, or `nil` to
        /// estimate one from ``tolerance``.
        ///
        /// Each non-equispaced position uses `2 * windowHalfWidth` grid points.
        public let windowHalfWidth: Int?

        /// Creates an NFFT configuration.
        ///
        /// - Parameters:
        ///   - oversamplingFactor: The integer FFT-grid size multiplier. Values
        ///     of at least two are supported.
        ///   - tolerance: The truncation-tolerance parameter. It must be finite
        ///     and strictly between zero and one.
        ///   - windowHalfWidth: An optional positive explicit half-width. If
        ///     omitted, the selected kernel estimates it from `tolerance`.
        @inlinable
        public init(
            oversamplingFactor: Int = 3,
            tolerance: Double = 1e-8,
            windowHalfWidth: Int? = nil
        ) {
            precondition(
                oversamplingFactor >= 2,
                "The NFFT oversampling factor must be at least two."
            )
            precondition(
                tolerance.isFinite && tolerance > 0.0 && tolerance < 1.0,
                "The NFFT tolerance must be finite and strictly between zero and one."
            )
            if let windowHalfWidth {
                precondition(
                    windowHalfWidth > 0,
                    "The NFFT window half-width must be positive."
                )
            }

            self.oversamplingFactor = oversamplingFactor
            self.tolerance = tolerance
            self.windowHalfWidth = windowHalfWidth
        }
    }
}
