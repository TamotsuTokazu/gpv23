#include <benchmark/benchmark.h>

#include "ntt.h"
#include "dcrtpoly.h"
#include "rlwe.h"

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

static void BM_BaseExtend(benchmark::State& state) {
    using NTT1 = NTT<1125898445242369LL, 13LL, 12289, 11>;
    using NTT2 = NTT<1125893310996481LL, 7LL, 12289, 11>;
    using NTT3 = NTT<1125893159989249LL, 13LL, 12289, 11>;

    using DCRT = DCRTPoly<NTT1, NTT2, NTT3>;
    Poly<12288> a0{1, 2, 3, 4, 5, 6};
    DCRT a{a0};
    for (auto _ : state) {
        DCRT a1 = a.BaseExtend<NTT1>();
        DCRT a2 = a.BaseExtend<NTT2>();
        DCRT a3 = a.BaseExtend<NTT3>();
    }
}

BENCHMARK(BM_BaseExtend)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_ExtMult(benchmark::State& state) {
    using NTT1 = NTT<1125898445242369LL, 13LL, 12289, 11>;
    using NTT2 = NTT<1125893310996481LL, 7LL, 12289, 11>;
    using NTT3 = NTT<1125893159989249LL, 13LL, 12289, 11>;

    using DCRT = DCRTPoly<NTT1, NTT2, NTT3>;
    Poly<12288> a0{1, 2, 3, 4, 5, 6};
    DCRT a{a0};
    RLWECiphertext<DCRT> c = RLWEEncrypt(a, a);
    RGSWCiphertext<DCRT> C = RGSWEncrypt(a, a);
    for (auto _ : state) {
        RLWECiphertext<DCRT> res = ExtMult(c, C);
    }
}

BENCHMARK(BM_ExtMult)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

// Register the benchmark main function if not already provided
// This is typically handled by Google Benchmark's CMake integration
// but can be added here if necessary.

// BENCHMARK_MAIN(); // Uncomment if needed
