# sebbu-fft

`sebbu-fft` is a small Swift package for discrete Fourier
transforms of complex, double-precision data. It provides a backend-neutral API,
portable Swift implementations, optional Accelerate-backed plans and a minimal
plan protocol that other packages can adopt.

The package is deliberately focused: applications can depend on one stable FFT
interface without making a native FFT library part of the core dependency graph.

## Features

- Forward and inverse transforms for `[Complex<Double>]`
- Unnormalized forward transforms and `1 / N`-normalized inverse transforms
- Angular-frequency axes derived from the input spacing
- In-place `fftShift` and `ifftShift` operations for arrays and mutable spans
- Reusable plans for repeated, fixed-size transforms
- A pure-Swift radix-2 implementation with Bluestein fallback for other sizes
- Conditional Accelerate-backed plan types

## Installation

Add the package in `Package.swift`:

```swift
dependencies: [
    .package(
        url: "https://github.com/MarSe32m/sebbu-fft.git",
        from: "0.2.0"
    )
]
```

Then add the library product to your target:

```swift
.target(
    name: "MyTarget",
    dependencies: [
        .product(name: "SebbuFFT", package: "sebbu-fft")
    ]
)
```

## Quick start

```swift
import ComplexModule
import SebbuFFT

let samples: [Complex<Double>] = [
    Complex(1, 0),
    Complex(0, 0),
    Complex(-1, 0),
    Complex(0, 0),
]

let sampleSpacing = 0.001 // seconds
let (angularFrequencies, spectrum) = FFT.fft(
    samples,
    spacing: sampleSpacing
)
```

`angularFrequencies` is measured in radians per second because the input spacing
is measured in seconds. The bins are returned in standard unshifted FFT order:
zero and positive frequencies first, followed by negative frequencies.

To center the zero-frequency bin for display or analysis:

```swift
var centeredFrequencies = angularFrequencies
var centeredSpectrum = spectrum

FFT.fftShift(
    frequencies: &centeredFrequencies,
    spectrum: &centeredSpectrum
)

// Restore the order expected by the inverse transform.
FFT.ifftShift(&centeredFrequencies, &centeredSpectrum)
```

For a round trip, pass the angular-frequency bin spacing to the inverse transform:

```swift
let angularFrequencySpacing =
    2 * Double.pi / (Double(samples.count) * sampleSpacing)

let (samplePositions, reconstructed) = FFT.ifft(
    spectrum,
    spacing: angularFrequencySpacing
)
```

Apart from floating-point error, `reconstructed` contains the original samples.
`samplePositions` has the same spacing as the original sample axis.

## Transform conventions

For `N` samples, the forward transform is

```text
X[k] = sum(x[n] * exp(-i * 2π * n * k / N), n = 0 ..< N)
```

and the inverse transform is

```text
x[n] = (1 / N) * sum(X[k] * exp(i * 2π * n * k / N), k = 0 ..< N)
```

| Operation | `spacing` | Returned axis | Scaling |
| --- | --- | --- | --- |
| `FFT.fft` | Sample spacing `Δt` | `ωₖ = 2πk / (NΔt)` | None |
| `FFT.ifft` | Angular bin spacing `Δω` | `tₙ = 2πn / (NΔω)` | `1 / N` |

Both operations use unshifted bin order. Call `fftShift` only when a centered
ordering is useful and restore the original order with `ifftShift` before an
inverse transform.

## Reusing plans

The one-shot functions create a default plan for each call. For a sequence of
same-sized transforms, create the plan once and reuse it:

```swift
let plan = FFT.defaultFFTPlan(sampleSize: samples.count)

for block in blocks {
    precondition(block.count == samples.count)
    let (frequencies, spectrum) = plan.execute(
        block,
        spacing: sampleSpacing
    )
    // Consume this block's spectrum.
}
```

Use `FFT.defaultiFFTPlan(sampleSize:)` for inverse transforms. A plan should be
executed only with arrays whose count matches the size used to create it. This is
particularly important for native backends, which commonly allocate fixed-size
state during initialization.

The package also exposes concrete plans when an application wants to choose a
backend explicitly:

- `FFT.FFTPlan` and `FFT.iFFTPlan` are portable Swift implementations.
- `FFT.AccelerateFFTPlan` and `FFT.AccelerateiFFTPlan` are available when
  Accelerate can be imported.

Backend choice made by the `default` factories is an implementation detail. Code
that requires a particular backend should construct that backend's plan directly.

## Compatible sample sizes

Use `FFT.isCompatibleSampleSize(_:)` to query the sizes preferred by the default
backend or use `FFT.nextCompatibleSampleSize(_:)` when choosing a padded length:

```swift
let requestedCount = max(input.count, 8)
let paddedCount = FFT.nextCompatibleSampleSize(requestedCount)
```

On macOS, preferred sizes are at least eight and have the form `f * 2^k`, where
`f` is `1`, `3`, `5`, or `15`. Other platforms currently accept every size of at
least eight as compatible with the portable default. Explicit backends may have
different constraints.

## Adding another backend

`FFT.Plan` is the extension point for optimized implementations. It intentionally
contains one operation:

```swift
func execute(
    _ input: [Complex<Double>],
    spacing: Double
) -> (x: [Double], y: [Complex<Double>])
```

A conforming plan should:

1. Own and release any backend-specific planning resources.
2. Require `x.count` to match the plan's configured sample size.
3. Return exactly one axis value and one transformed value per input element.
4. Use an unnormalized negative-exponent forward transform or a `1 / N`-normalized
   positive-exponent inverse transform.
5. Return forward angular frequencies or inverse sample positions using the spacing
   conventions above.
6. Return bins in unshifted order.
7. Document whether an instance can be executed concurrently.

Separate forward and inverse plan types are recommended because the protocol does
not encode transform direction.

## Numerical and API notes

- All transforms operate on `Complex<Double>` values.
- The APIs allocate output arrays; shift operations mutate existing storage in place.
- `spacing` should be finite and greater than zero.
- Floating-point round trips are approximate.
- Plan reuse can avoid repeated setup costs, but execution-time allocation
  depends on the selected backend.
- Thread-safety is a property of each plan implementation, not of the protocol itself.

## License

`sebbu-fft` is available under the MIT license included in `LICENSE`.
