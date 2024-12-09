#include <benchmark/benchmark.h>

#include "ntt.h"
#include "dcrtpoly.h"
#include "rlwe.h"

#define REPETITIONS 3

static void BM_ForwardCT23NTT(benchmark::State& state) {
    using NTT = NTT<562936689020929LL, 7LL, 12289, 11>;
    alignas(64) uint64_t a[NTT::N] = {0, 1};
    alignas(64) uint64_t b[NTT::N];
    NTT::GetInstance().ForwardCT23NTT(a, b);
    for (auto _ : state) {
        a[rand() % NTT::N] = rand();
        NTT::GetInstance().ForwardCT23NTT(a, b);
    }
}

BENCHMARK(BM_ForwardCT23NTT)->Unit(benchmark::kMicrosecond)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_ForwardRaderNTT12289(benchmark::State& state) {
    using NTT = NTT<562936689020929LL, 19, 12289, 11>;
    uint64_t a[NTT::N] = {0, 1};
    for (auto _ : state) {
        a[rand() % NTT::N] = rand();
        NTT::GetInstance().ForwardNTT(a);
    }
}

BENCHMARK(BM_ForwardRaderNTT12289)->Unit(benchmark::kMicrosecond)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_BaseExtend(benchmark::State& state) {
    constexpr size_t p = 12289;
    constexpr size_t gp = 11;
    using NTT1 = CircNTT<562936689020929LL, 7LL, p, gp>;
    using NTT2 = CircNTT<562918719160321LL, 14LL, p, gp>;
    using NTT3 = CircNTT<562880212316161LL, 14LL, p, gp>;
    using NTT4 = CircNTT<562851973963777LL, 5LL, p, gp>;

    using DCRT = DCRTPoly<NTT1, NTT2, NTT3, NTT4>;
    for (auto _ : state) {
        DCRT a = DCRT::SampleUniform();
        DCRT a1 = a.BaseExtend<NTT1>();
    }
}

BENCHMARK(BM_BaseExtend)->Unit(benchmark::kMicrosecond)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

static void BM_ExtMult(benchmark::State& state) {
    constexpr size_t p = 12289;
    constexpr size_t gp = 11;
    using NTT1 = CircNTT<562936689020929LL, 7LL, p, gp>;
    using NTT2 = CircNTT<562918719160321LL, 14LL, p, gp>;
    using NTT3 = CircNTT<562880212316161LL, 14LL, p, gp>;
    using NTT4 = CircNTT<562851973963777LL, 5LL, p, gp>;

    using DCRT = DCRTPoly<NTT1, NTT2, NTT3, NTT4>;
    Poly<p> a0{1, 2, 3, 4, 5, 6};
    DCRT a{a0};
    RGSWCiphertext<DCRT> C = RGSWEncrypt(a, a);
    for (auto _ : state) {
        RLWECiphertext<DCRT> c = RLWEEncrypt(DCRT::SampleUniform(), a);
        RLWECiphertext<DCRT> res = ExtMult(c, C);
    }
}

BENCHMARK(BM_ExtMult)->Unit(benchmark::kMicrosecond)->Repetitions(REPETITIONS)->ReportAggregatesOnly(true);

// Register the benchmark main function if not already provided
// This is typically handled by Google Benchmark's CMake integration
// but can be added here if necessary.

// BENCHMARK_MAIN(); // Uncomment if needed
