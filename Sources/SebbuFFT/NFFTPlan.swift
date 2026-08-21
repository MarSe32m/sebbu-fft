// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import RealModule

public extension FFT {
    /// A reusable one-dimensional non-equispaced forward-transform plan.
    ///
    /// The plan precomputes the Gaussian-window deconvolution factors and the
    /// sparse interpolation stencil for a fixed coefficient count and fixed set
    /// of positions. Execution embeds the centered coefficients in an
    /// oversampled grid, delegates the regular transform to an ``FFT/Plan``, and
    /// interpolates the grid values at the requested positions.
    ///
    /// Plan instances should not be executed concurrently unless the selected
    /// underlying FFT plan documents that concurrent execution is supported.
    struct NFFTPlan {
        /// The required number of centered Fourier coefficients.
        public let coefficientCount: Int

        /// The number of non-equispaced output positions.
        public let positionCount: Int

        /// The size of the oversampled regular FFT grid.
        public let gridSize: Int

        /// Half the number of grid points in each truncated interpolation
        /// stencil. Each position uses twice this many points.
        public let windowHalfWidth: Int
        
        @usableFromInline
        internal let precomputation: NFFTPrecomputation
        @usableFromInline
        internal let fftPlan: any Plan

        /// Creates a forward NFFT plan using the package's default FFT backend.
        ///
        /// - Parameters:
        ///   - positions: Finite evaluation positions measured in turns. Values
        ///     outside `[-0.5, 0.5)` are wrapped periodically.
        ///   - coefficientCount: The positive, even number of centered Fourier
        ///     coefficients passed to ``execute(_:)``.
        ///   - configuration: Oversampling and accuracy controls.
        ///   - kernel: The interpolation kernel.
        @inlinable
        public init(
            positions: [Double],
            coefficientCount: Int,
            configuration: NFFTConfiguration = .init(),
            kernel: any NFFTKernel = GaussianNFFTKernel()
        ) {
            let precomputation = NFFTPrecomputation(
                positions: positions,
                coefficientCount: coefficientCount,
                configuration: configuration,
                kernel: kernel
            )
            self.init(
                precomputation: precomputation,
                fftPlan: FFT.defaultFFTPlan(sampleSize: precomputation.gridSize)
            )
        }

        /// Creates a forward NFFT plan with an explicitly selected FFT plan.
        ///
        /// The supplied plan must accept arrays whose count equals
        /// `coefficientCount * configuration.oversamplingFactor`.
        @inlinable
        public init(
            positions: [Double],
            coefficientCount: Int,
            configuration: NFFTConfiguration = .init(),
            kernel: any NFFTKernel = GaussianNFFTKernel(),
            fftPlan: any Plan
        ) {
            let precomputation = NFFTPrecomputation(
                positions: positions,
                coefficientCount: coefficientCount,
                configuration: configuration,
                kernel: kernel
            )
            self.init(precomputation: precomputation, fftPlan: fftPlan)
        }
        
        @inlinable
        internal init(
            precomputation: NFFTPrecomputation,
            fftPlan: any Plan
        ) {
            self.coefficientCount = precomputation.coefficientCount
            self.positionCount = precomputation.positionCount
            self.gridSize = precomputation.gridSize
            self.windowHalfWidth = precomputation.windowHalfWidth
            self.precomputation = precomputation
            self.fftPlan = fftPlan
        }

        /// Evaluates a centered Fourier series at the plan's non-equispaced
        /// positions.
        ///
        /// - Parameter coefficients: Exactly ``coefficientCount`` coefficients
        ///   ordered from wave number `-N / 2` through `N / 2 - 1`.
        /// - Returns: One approximate Fourier-series value per plan position.
        @inlinable
        public func execute(
            _ coefficients: [Complex<Double>]
        ) -> [Complex<Double>] {
            precondition(
                coefficients.count == coefficientCount,
                "The coefficient count must match the NFFT plan."
            )
            guard positionCount > 0 else { return [] }

            var paddedCoefficients = [Complex<Double>](
                repeating: .zero,
                count: gridSize
            )
            let halfCount = coefficientCount / 2

            for coefficientIndex in 0..<halfCount {
                let gridIndex = gridSize - halfCount + coefficientIndex
                paddedCoefficients[gridIndex] = FFT.nfftScale(
                    coefficients[coefficientIndex],
                    by: precomputation.deconvolution[coefficientIndex]
                )
            }
            for coefficientIndex in halfCount..<coefficientCount {
                let gridIndex = coefficientIndex - halfCount
                paddedCoefficients[gridIndex] = FFT.nfftScale(
                    coefficients[coefficientIndex],
                    by: precomputation.deconvolution[coefficientIndex]
                )
            }

            let gridValues = fftPlan.execute(
                paddedCoefficients,
                spacing: 1.0
            ).y
            precondition(
                gridValues.count == gridSize,
                "The FFT plan returned an unexpected number of grid values."
            )

            var result = [Complex<Double>](
                repeating: .zero,
                count: positionCount
            )
            let stencilWidth = precomputation.stencilWidth
            for positionIndex in 0..<positionCount {
                let stencilStart = positionIndex * stencilWidth
                var value = Complex<Double>.zero
                for stencilIndex in stencilStart..<(stencilStart + stencilWidth) {
                    value += FFT.nfftScale(
                        gridValues[precomputation.gridIndices[stencilIndex]],
                        by: precomputation.weights[stencilIndex]
                    )
                }
                result[positionIndex] = value
            }

            return result
        }
    }

    /// A reusable plan for the adjoint one-dimensional NFFT.
    ///
    /// This plan is the numerical adjoint of ``NFFTPlan`` when both plans use
    /// the same positions, configuration, kernel, and compatible FFT backends.
    struct NFFTAdjointPlan {
        /// The number of centered Fourier coefficients returned by the plan.
        public let coefficientCount: Int

        /// The required number of values at non-equispaced positions.
        public let positionCount: Int

        /// The size of the oversampled regular FFT grid.
        public let gridSize: Int

        /// Half the number of grid points in each truncated interpolation
        /// stencil.
        public let windowHalfWidth: Int

        @usableFromInline
        internal let precomputation: NFFTPrecomputation
        @usableFromInline
        internal let inverseFFTPlan: any Plan

        /// Creates an adjoint NFFT plan using the package's default inverse FFT
        /// backend.
        @inlinable
        public init(
            positions: [Double],
            coefficientCount: Int,
            configuration: NFFTConfiguration = .init(),
            kernel: any NFFTKernel = GaussianNFFTKernel()
        ) {
            let precomputation = NFFTPrecomputation(
                positions: positions,
                coefficientCount: coefficientCount,
                configuration: configuration,
                kernel: kernel
            )
            self.init(
                precomputation: precomputation,
                inverseFFTPlan: FFT.defaultiFFTPlan(
                    sampleSize: precomputation.gridSize
                )
            )
        }

        /// Creates an adjoint NFFT plan with an explicitly selected inverse FFT
        /// plan.
        ///
        /// The supplied plan must implement the package's normalized inverse
        /// convention for arrays whose count equals the plan's ``gridSize``.
        @inlinable
        public init(
            positions: [Double],
            coefficientCount: Int,
            configuration: NFFTConfiguration = .init(),
            kernel: any NFFTKernel = GaussianNFFTKernel(),
            inverseFFTPlan: any Plan
        ) {
            let precomputation = NFFTPrecomputation(
                positions: positions,
                coefficientCount: coefficientCount,
                configuration: configuration,
                kernel: kernel
            )
            self.init(
                precomputation: precomputation,
                inverseFFTPlan: inverseFFTPlan
            )
        }
        
        @inlinable
        internal init(
            precomputation: NFFTPrecomputation,
            inverseFFTPlan: any Plan
        ) {
            self.coefficientCount = precomputation.coefficientCount
            self.positionCount = precomputation.positionCount
            self.gridSize = precomputation.gridSize
            self.windowHalfWidth = precomputation.windowHalfWidth
            self.precomputation = precomputation
            self.inverseFFTPlan = inverseFFTPlan
        }

        /// Applies the adjoint transform to values at the plan's positions.
        ///
        /// - Parameter values: Exactly ``positionCount`` complex values.
        /// - Returns: ``coefficientCount`` values in centered wave-number order.
        @inlinable
        public func execute(_ values: [Complex<Double>]) -> [Complex<Double>] {
            precondition(
                values.count == positionCount,
                "The NFFT positions and values must have the same count."
            )
            guard positionCount > 0 else {
                return [Complex<Double>](
                    repeating: .zero,
                    count: coefficientCount
                )
            }

            var gridValues = [Complex<Double>](
                repeating: .zero,
                count: gridSize
            )
            let stencilWidth = precomputation.stencilWidth
            for positionIndex in 0..<positionCount {
                let stencilStart = positionIndex * stencilWidth
                for stencilIndex in stencilStart..<(stencilStart + stencilWidth) {
                    let gridIndex = precomputation.gridIndices[stencilIndex]
                    gridValues[gridIndex] += FFT.nfftScale(
                        values[positionIndex],
                        by: precomputation.weights[stencilIndex]
                    )
                }
            }

            let inverseGrid = inverseFFTPlan.execute(
                gridValues,
                spacing: 1.0
            ).y
            precondition(
                inverseGrid.count == gridSize,
                "The inverse FFT plan returned an unexpected number of grid values."
            )

            var coefficients = [Complex<Double>](
                repeating: .zero,
                count: coefficientCount
            )
            let halfCount = coefficientCount / 2
            for coefficientIndex in 0..<halfCount {
                let gridIndex = gridSize - halfCount + coefficientIndex
                let scale = Double(gridSize)
                    * precomputation.deconvolution[coefficientIndex]
                coefficients[coefficientIndex] = FFT.nfftScale(
                    inverseGrid[gridIndex],
                    by: scale
                )
            }
            for coefficientIndex in halfCount..<coefficientCount {
                let gridIndex = coefficientIndex - halfCount
                let scale = Double(gridSize)
                    * precomputation.deconvolution[coefficientIndex]
                coefficients[coefficientIndex] = FFT.nfftScale(
                    inverseGrid[gridIndex],
                    by: scale
                )
            }

            return coefficients
        }
    }
}

@usableFromInline
internal struct NFFTPrecomputation {
    @usableFromInline
    let coefficientCount: Int
    @usableFromInline
    let positionCount: Int
    @usableFromInline
    let gridSize: Int
    @usableFromInline
    let windowHalfWidth: Int
    @usableFromInline
    let stencilWidth: Int
    @usableFromInline
    let deconvolution: [Double]
    @usableFromInline
    let gridIndices: [Int]
    @usableFromInline
    let weights: [Double]
    
    @inlinable
    init(
        positions: [Double],
        coefficientCount: Int,
        configuration: FFT.NFFTConfiguration,
        kernel: any FFT.NFFTKernel
    ) {
        FFT.preconditionValidNFFTCoefficientCount(coefficientCount)
        FFT.preconditionFiniteNFFTPositions(positions)

        let (gridSize, gridSizeOverflow) = coefficientCount
            .multipliedReportingOverflow(by: configuration.oversamplingFactor)
        precondition(
            !gridSizeOverflow,
            "The oversampled NFFT grid size is not representable as an Int."
        )

        let windowHalfWidth = configuration.windowHalfWidth
            ?? kernel.estimateWindowHalfWidth(
                tolerance: configuration.tolerance,
                coefficientCount: coefficientCount,
                oversamplingFactor: configuration.oversamplingFactor
            )
        precondition(
            windowHalfWidth > 0 && windowHalfWidth <= gridSize / 2,
            "The NFFT window half-width must be positive and no greater than half the grid size."
        )

        let stencilWidth = 2 * windowHalfWidth
        let (stencilCount, stencilCountOverflow) = positions.count
            .multipliedReportingOverflow(by: stencilWidth)
        precondition(
            !stencilCountOverflow,
            "The NFFT interpolation stencil size is not representable as an Int."
        )

        let halfCount = coefficientCount / 2
        var deconvolution = [Double]()
        deconvolution.reserveCapacity(coefficientCount)
        for coefficientIndex in 0..<coefficientCount {
            let waveNumber = coefficientIndex - halfCount
            let fourierValue = kernel.fourierValue(
                at: waveNumber,
                gridSize: gridSize,
                windowHalfWidth: windowHalfWidth,
                oversamplingFactor: configuration.oversamplingFactor
            )
            let denominator = fourierValue * Double(gridSize)
            precondition(
                denominator.isFinite && denominator != 0.0,
                "The NFFT kernel has a zero or non-finite Fourier value in the requested band."
            )
            let correction = 1.0 / denominator
            precondition(
                correction.isFinite,
                "The NFFT kernel deconvolution factor is not finite."
            )
            deconvolution.append(correction)
        }

        var gridIndices = [Int]()
        var weights = [Double]()
        gridIndices.reserveCapacity(stencilCount)
        weights.reserveCapacity(stencilCount)
        for position in positions {
            let wrappedPosition = FFT.nfftWrappedPosition(position)
            let scaledPosition = Double(gridSize) * wrappedPosition
            let baseGridIndex = Int(scaledPosition.rounded(.down))
            for offset in -windowHalfWidth..<windowHalfWidth {
                let unwrappedGridIndex = baseGridIndex + offset
                let gridPosition = Double(unwrappedGridIndex) / Double(gridSize)
                let distance = FFT.nfftWrappedPosition(
                    wrappedPosition - gridPosition
                )
                let weight = kernel.value(
                    at: distance,
                    gridSize: gridSize,
                    windowHalfWidth: windowHalfWidth,
                    oversamplingFactor: configuration.oversamplingFactor
                )
                precondition(
                    weight.isFinite,
                    "The NFFT kernel returned a non-finite interpolation weight."
                )
                gridIndices.append(
                    FFT.nfftPositiveModulo(unwrappedGridIndex, modulus: gridSize)
                )
                weights.append(weight)
            }
        }

        self.coefficientCount = coefficientCount
        self.positionCount = positions.count
        self.gridSize = gridSize
        self.windowHalfWidth = windowHalfWidth
        self.stencilWidth = stencilWidth
        self.deconvolution = deconvolution
        self.gridIndices = gridIndices
        self.weights = weights
    }
}
