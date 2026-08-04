// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

import ComplexModule
import SebbuFFT
import Testing

private let shiftSampleCounts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

private func expectedFFTShift<Element>(_ values: [Element]) -> [Element] {
    let splitIndex = (values.count + 1) / 2
    return Array(values.dropFirst(splitIndex))
        + Array(values.prefix(splitIndex))
}

private func taggedValues(
    count: Int
) -> (frequencies: [Double], spectrum: [Complex<Double>]) {
    let frequencies = (0..<count).map(Double.init)
    let spectrum = frequencies.map { Complex($0, -$0) }
    return (frequencies, spectrum)
}

@Suite("FFT shifts")
struct ShiftTests {
    @Test("fftShift uses the standard centered permutation", arguments: shiftSampleCounts)
    func fftShiftPermutation(sampleCount: Int) {
        let original = taggedValues(count: sampleCount)
        var frequencies = original.frequencies
        var spectrum = original.spectrum

        FFT.fftShift(frequencies: &frequencies, spectrum: &spectrum)

        #expect(frequencies == expectedFFTShift(original.frequencies))
        #expect(spectrum == expectedFFTShift(original.spectrum))

        for index in frequencies.indices {
            #expect(spectrum[index].real == frequencies[index])
            #expect(spectrum[index].imaginary == -frequencies[index])
        }
    }

    @Test("ifftShift reverses fftShift", arguments: shiftSampleCounts)
    func shiftsRoundTrip(sampleCount: Int) {
        let original = taggedValues(count: sampleCount)
        var frequencies = original.frequencies
        var spectrum = original.spectrum

        FFT.fftShift(frequencies: &frequencies, spectrum: &spectrum)
        FFT.ifftShift(&frequencies, &spectrum)

        #expect(frequencies == original.frequencies)
        #expect(spectrum == original.spectrum)
    }

    @Test("Array and MutableSpan overloads agree", arguments: shiftSampleCounts)
    func arrayAndSpanOverloadsAgree(sampleCount: Int) {
        let original = taggedValues(count: sampleCount)
        var arrayFrequencies = original.frequencies
        var arraySpectrum = original.spectrum
        var spanFrequencies = original.frequencies
        var spanSpectrum = original.spectrum

        FFT.fftShift(
            frequencies: &arrayFrequencies,
            spectrum: &arraySpectrum
        )
        do {
            var frequencySpan = spanFrequencies.mutableSpan
            var spectrumSpan = spanSpectrum.mutableSpan
            FFT.fftShift(&frequencySpan, &spectrumSpan)
        }

        #expect(spanFrequencies == arrayFrequencies)
        #expect(spanSpectrum == arraySpectrum)

        FFT.ifftShift(&arrayFrequencies, &arraySpectrum)
        do {
            var frequencySpan = spanFrequencies.mutableSpan
            var spectrumSpan = spanSpectrum.mutableSpan
            FFT.ifftShift(&frequencySpan, &spectrumSpan)
        }

        #expect(spanFrequencies == arrayFrequencies)
        #expect(spanSpectrum == arraySpectrum)
        #expect(spanFrequencies == original.frequencies)
        #expect(spanSpectrum == original.spectrum)
    }
}