// Copyright (c) 2026 Sebastian Toivonen
// SPDX-License-Identifier: MIT

#if canImport(Accelerate)
import Accelerate
import ComplexModule
import SebbuFFT
import Testing

private let accelerateSampleCounts = [8, 10, 12, 15, 16, 20, 24, 30, 32]

@Suite("Accelerate plans")
struct AcceleratePlanTests {
    @Test("Accelerate forward plan matches the independent DFT", arguments: accelerateSampleCounts)
    func forwardMatchesNaiveDFT(sampleCount: Int) {
        let input = deterministicSignal(count: sampleCount, seed: 10)
        let plan = FFT.AccelerateFFTPlan(sampleSize: sampleCount)
        expectForwardPlanMatchesOracle(plan, input: input, spacing: 0.125)
    }

    @Test("Accelerate inverse plan matches the independent DFT", arguments: accelerateSampleCounts)
    func inverseMatchesNaiveDFT(sampleCount: Int) {
        let input = deterministicSignal(count: sampleCount, seed: 11)
        let plan = FFT.AccelerateiFFTPlan(sampleSize: sampleCount)
        expectInversePlanMatchesOracle(plan, input: input, spacing: 0.375)
    }

    @Test("An Accelerate plan can be reused")
    func planReuse() {
        let sampleCount = 16
        let forwardPlan = FFT.AccelerateFFTPlan(sampleSize: sampleCount)
        let inversePlan = FFT.AccelerateiFFTPlan(sampleSize: sampleCount)

        for seed in 12..<18 {
            let input = deterministicSignal(count: sampleCount, seed: seed)
            expectForwardPlanMatchesOracle(
                forwardPlan,
                input: input,
                spacing: 0.25
            )
            expectInversePlanMatchesOracle(
                inversePlan,
                input: input,
                spacing: 0.5
            )
        }
    }
}
#endif