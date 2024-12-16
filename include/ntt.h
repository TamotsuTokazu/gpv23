#ifndef NTT_H
#define NTT_H

#include <cstdint>
#include <cstddef>
#include <array>
#include <iostream>

#include <immintrin.h>
#include <hexl/hexl.hpp>

#include "zp.h"

#include "nttavx512ifma.h"
#include "nttavx512dq.h"
#include "nttscalar.h"

#if defined(HAS_AVX512IFMA)

template <uint64_t p_, uint64_t g_, size_t O_, size_t w_>
using NTT23 = NTT23AVX512IFMA<p_, g_, O_, w_>;

#elif defined(HAS_AVX512DQ)

template <uint64_t p_, uint64_t g_, size_t O_, size_t w_>
using NTT23 = NTT23AVX512DQ<p_, g_, O_, w_>;

#else

template <uint64_t p_, uint64_t g_, size_t O_, size_t w_>
using NTT23 = NTT23Scalar<p_, g_, O_, w_>;

#endif

template <typename NTT_> requires (NTT_::N == NTT_::O - 1)
class CircNTT {
public:
    using PrimitiveNTT = NTT_;

    constexpr static uint64_t p = PrimitiveNTT::p;
    constexpr static uint64_t g = PrimitiveNTT::g;
    constexpr static size_t O = PrimitiveNTT::O;
    constexpr static size_t w = PrimitiveNTT::w;
    constexpr static size_t N = PrimitiveNTT::O;

    using Z = Zp<p>;

    constexpr static uint64_t N_inv = Z::Inv(N);

    inline void ForwardNTT(uint64_t a[]) {
        auto t = a[0];
        for (size_t i = 1; i < N; i++) {
            a[0] = Z::Add(a[0], a[i]);
        }
        PrimitiveNTT::GetInstance().ForwardNTT(a + 1);
        intel::hexl::EltwiseAddMod(a + 1, a + 1, t, N - 1, p);
    }

    inline void InverseNTT(uint64_t a[]) {
        PrimitiveNTT::GetInstance().InverseNTT(a + 1);
        auto t = a[0];
        for (size_t i = 1; i < N; i++) {
            t = Z::Sub(t, a[i]);
        }
        a[0] = Z::Mul(t, N_inv);
        intel::hexl::EltwiseAddMod(a + 1, a + 1, a[0], N - 1, p);
    }

    static CircNTT &GetInstance() {
        static CircNTT instance;
        return instance;
    }

private:
    CircNTT() {}
};

template <typename NTTp_, typename NTTq_>
class TensorNTTImpl {
public:
    using NTTp = NTTp_;
    using NTTq = NTTq_;

    constexpr static size_t N = NTTp::N * NTTq::N;
    constexpr static size_t O = NTTp::O * NTTq::O;

    static_assert(NTTp::p == NTTq::p, "p must be the same");
    constexpr static uint64_t p = NTTp::p;

    using Z = Zp<p>;

    static_assert(NTTp::g == NTTq::g, "g must be the same");

    static void ForwardNTT(uint64_t a[]) {
        static uint64_t reg[N];

        for (size_t i = 0; i < NTTp::N; i++) {
            auto *b = reg + NTTq::N * i;
            for (size_t j = 0; j < NTTq::N; j++) {
                b[j] = a[NTTq::N * i + j];
            }
            NTTq::GetInstance().ForwardNTT(b);
        }

        std::copy(reg, reg + N, a);

        for (size_t j = 0; j < NTTq::N; j++) {
            auto *b = reg + NTTp::N * j;
            for (size_t i = 0; i < NTTp::N; i++) {
                b[i] = a[NTTq::N * i + j];
            }
            NTTp::GetInstance().ForwardNTT(b);
        }
        for (size_t j = 0; j < NTTq::N; j++) {
            for (size_t i = 0; i < NTTp::N; i++) {
                a[NTTq::N * i + j] = reg[NTTp::N * j + i];
            }
        }
    }

    static void InverseNTT(uint64_t a[]) {
        static uint64_t reg[N];

        for (size_t j = 0; j < NTTq::N; j++) {
            auto *b = reg + NTTp::N * j;
            for (size_t i = 0; i < NTTp::N; i++) {
                b[i] = a[NTTq::N * i + j];
            }
            NTTp::GetInstance().InverseNTT(b);
        }
        for (size_t j = 0; j < NTTq::N; j++) {
            for (size_t i = 0; i < NTTp::N; i++) {
                a[NTTq::N * i + j] = reg[NTTp::N * j + i];
            }
        }

        for (size_t i = 0; i < NTTp::N; i++) {
            auto *b = reg + NTTq::N * i;
            for (size_t j = 0; j < NTTq::N; j++) {
                b[j] = a[NTTq::N * i + j];
            }
            NTTq::GetInstance().InverseNTT(b);
        }

        std::copy(reg, reg + N, a);
    }

    static TensorNTTImpl &GetInstance() {
        static TensorNTTImpl instance;
        return instance;
    }

private:
    TensorNTTImpl() {}
};

#endif // NTT_H