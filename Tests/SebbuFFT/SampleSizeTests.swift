import SebbuFFT
import Testing

@Suite("Compatible sample sizes")
struct SampleSizeTests {
    @Test("Counts below eight are not compatible")
    func countsBelowMinimumAreIncompatible() {
        for sampleCount in -16..<8 {
            #expect(!FFT.isCompatibleSampleSize(sampleCount))
        }
    }

    @Test("nextCompatibleSampleSize returns eight below the minimum")
    func requestsBelowMinimumReturnEight() {
        for requestedCount in -16..<8 {
            #expect(FFT.nextCompatibleSampleSize(requestedCount) == 8)
        }
    }

    @Test("Known platform-specific sizes are classified correctly")
    func knownCompatibilityValues() {
        #if os(macOS)
        let compatible = [8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 64]
        let incompatible = [9, 11, 13, 14, 17, 18, 19, 21, 22, 23, 25, 26, 27]

        for sampleCount in compatible {
            #expect(FFT.isCompatibleSampleSize(sampleCount))
        }
        for sampleCount in incompatible {
            #expect(!FFT.isCompatibleSampleSize(sampleCount))
        }
        #else
        for sampleCount in 8...256 {
            #expect(FFT.isCompatibleSampleSize(sampleCount))
        }
        #endif
    }

    @Test("nextCompatibleSampleSize returns a compatible minimum")
    func nextCompatibleSizeIsMinimal() {
        for requestedCount in 8...4_096 {
            let result = FFT.nextCompatibleSampleSize(requestedCount)

            #expect(result >= requestedCount)
            #expect(FFT.isCompatibleSampleSize(result))
            for smallerCandidate in requestedCount..<result {
                #expect(!FFT.isCompatibleSampleSize(smallerCandidate))
            }
        }
    }

    @Test("nextCompatibleSampleSize chooses mixed-radix sizes on macOS")
    func mixedRadixRegressions() {
        #if os(macOS)
        let cases = [
            (requested: 8, expected: 8),
            (requested: 9, expected: 10),
            (requested: 11, expected: 12),
            (requested: 13, expected: 15),
            (requested: 14, expected: 15),
            (requested: 17, expected: 20),
            (requested: 25, expected: 30),
        ]

        for testCase in cases {
            #expect(
                FFT.nextCompatibleSampleSize(testCase.requested)
                    == testCase.expected
            )
        }
        #endif
    }

    @Test("nextCompatibleSampleSize is monotonic")
    func nextCompatibleSizeIsMonotonic() {
        var previous = FFT.nextCompatibleSampleSize(8)
        for requestedCount in 9...4_096 {
            let current = FFT.nextCompatibleSampleSize(requestedCount)
            #expect(current >= previous)
            previous = current
        }
    }
}