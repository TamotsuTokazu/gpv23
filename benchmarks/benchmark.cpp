#include <benchmark/benchmark.h>

#include "ntt.h"

#define REPETITIONS 3

static void BM_ForwardCT23NTT(benchmark::State& state) {
    using NTT = NTT<1152921504107839489LL, 19, 12289, 11>;
    uint64_t a[12288] = {0, 1};
    uint64_t b[12288];
    for (auto _ : state) {
        NTT::GetInstance().ForwardCT23NTT(a, b);
    }
}

BENCHMARK(BM_ForwardCT23NTT)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_ForwardRaderNTT769(benchmark::State& state) {
    using NTT = NTT<1152921504602791681LL, 11, 769, 11>;
    uint64_t a[768] = {0, 1};
    for (auto _ : state) {
        NTT::GetInstance().ForwardNTT(a);
    }
}

BENCHMARK(BM_ForwardRaderNTT769)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_ForwardRaderNTT12289(benchmark::State& state) {
    using NTT = NTT<1152921504107839489LL, 19, 12289, 11>;
    uint64_t a[12288] = {0, 1};
    for (auto _ : state) {
        NTT::GetInstance().ForwardNTT(a);
    }
}

BENCHMARK(BM_ForwardRaderNTT12289)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

// Register the benchmark main function if not already provided
// This is typically handled by Google Benchmark's CMake integration
// but can be added here if necessary.

// BENCHMARK_MAIN(); // Uncomment if needed
