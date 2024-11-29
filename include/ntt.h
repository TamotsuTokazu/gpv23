#ifndef NTT_H
#define NTT_H

#include <cstdint>
#include <cstddef>
#include <array>
#include <iostream>

#include <immintrin.h>

#include "zp.h"

template <uint64_t p_, uint64_t g_, size_t O_, size_t w_>
class NTT {
public:
    constexpr static uint64_t p = p_;
    constexpr static uint64_t g = g_;
    constexpr static size_t O = O_;
    constexpr static size_t w = w_;

    using Z = Zp<p>;

    static_assert(p % ((u_int64_t)O * (O - 1)) == 1, "p must be 1 mod O * (O - 1)");

    constexpr static size_t CompileComputeu(size_t N_) {
        size_t u = 0;
        while (N_ % 2 == 0) {
            N_ >>= 1;
            u++;
        }
        return u;
    }

    constexpr static size_t CompileComputeU(size_t N_) {
        size_t U = 1 << CompileComputeu(N_);
        return U;
    }

    constexpr static size_t CompileComputev(size_t N_) {
        size_t v = 0;
        while (N_ % 3 == 0) {
            N_ /= 3;
            v++;
        }
        return v;
    }

    constexpr static size_t CompileComputeV(size_t N_) {
        size_t V = 1;
        for (size_t i = 0; i < CompileComputev(N_); i++) {
            V *= 3;
        }
        return V;
    }

    constexpr static uint64_t ComputeOmegaO() {
        return Z::Pow(g, (p - 1) / O);
    }

    constexpr static uint64_t ComputeOmegaOInv() {
        return Z::Pow(ComputeOmegaO(), p - 2);
    }

    constexpr static uint64_t ComputeOmegaN() {
        return Z::Pow(g, (p - 1) / N);
    }

    constexpr static uint64_t ComputeOmegaNInv() {
        return Z::Pow(ComputeOmegaN(), p - 2);
    }

    constexpr static size_t N = O - 1;
    constexpr static size_t u = CompileComputeu(N);
    constexpr static size_t U = CompileComputeU(N);
    constexpr static size_t v = CompileComputev(N);
    constexpr static size_t V = CompileComputeV(N);

    constexpr static uint64_t omega_O = ComputeOmegaO();
    constexpr static uint64_t omega_O_inv = ComputeOmegaOInv();
    constexpr static uint64_t omega_N = ComputeOmegaN();
    constexpr static uint64_t omega_N_inv = ComputeOmegaNInv(); 

    constexpr static std::array<size_t, N> PrecomputeGi() {
        std::array<size_t, N> gi;
        gi[0] = 1;
        for (size_t i = 1; i < N; i++) {
            gi[i] = (uint64_t)gi[i - 1] * w % O;
        }
        return gi;
    }

    constexpr static std::array<size_t, O> PrecomputeGiInv() {
        std::array<size_t, O> gi_inv;
        gi_inv[0] = -1;
        for (size_t i = 0; i < N; i++) {
            gi_inv[gi[i]] = i;
        }
        return gi_inv;
    }

    constexpr static std::array<size_t, N> gi = PrecomputeGi();
    constexpr static std::array<size_t, O> gi_inv = PrecomputeGiInv();

    constexpr static size_t BitReverse(size_t x, size_t l, size_t r) {
        size_t y = 0;
        for (size_t i = 0; i < l; i++) {
            y = y * r + x % r;
            x /= r;
        }
        return y;
    }

    constexpr static std::array<uint64_t, N> PrecomputeBitReverseTable() {
        std::array<uint64_t, N> bit_reverse_table;
        for (size_t i = 0; i < V; i++) {
            for (size_t j = 0; j < U; j++) {
                bit_reverse_table[i * U + j] = BitReverse(j, u, 2) * V + BitReverse(i, v, 3);
            }
        }
        return bit_reverse_table;
    }

    constexpr static std::array<uint64_t, N> bit_reverse_table = PrecomputeBitReverseTable();

    
    static uint64_t omega_N_table[N];
    static uint64_t omega_N_inv_table[N];
    static uint64_t omega_N_barrett_table[N];
    static uint64_t omega_N_inv_barrett_table[N];

    static void ComputeOmegaNTable() {
        omega_N_table[0] = 1;
        for (size_t i = 1; i < N; i++) {
            omega_N_table[i] = Z::Mul(omega_N_table[i - 1], omega_N);
        }
        omega_N_inv_table[0] = 1;
        for (size_t i = 1; i < N; i++) {
            omega_N_inv_table[i] = omega_N_table[N - i];
        }
        for (size_t i = 0; i < N; i++) {
            omega_N_barrett_table[i] = Z::ComputeBarrettFactor(omega_N_table[i]);
            omega_N_inv_barrett_table[i] = Z::ComputeBarrettFactor(omega_N_inv_table[i]);
        }
    }

    void CT23NTT(uint64_t __restrict__ a[], uint64_t __restrict__ b[], uint64_t __restrict__ omega[], uint64_t __restrict__ omega_barrett[]) {

        for (size_t i = 0; i < N; i++) {
            b[i] = a[bit_reverse_table[i]];
        }

        for (size_t i = 0; i < N; i += 2) {
            uint64_t t = b[i + 1];
            b[i + 1] = Z::Sub(b[i], t);
            b[i] = Z::Add(b[i], t);
        }

        size_t l0 = 1, l1 = 2, d = N / 2;

        __m256i pV = _mm256_set1_epi64x(p);
        __m256i p_1V = _mm256_set1_epi64x(p - 1);

        for (size_t i = 1; i < u; i++) {
            l0 = 1 << i;
            l1 = 1 << (i + 1);
            d = N >> (i + 1);
            for (size_t j = 0; j < N; j += l1) {
                size_t k = 0;
                for (; k + 3 < l0; k += 4) {
                    uint64_t t0 = Z::MulFastConst(b[j + l0 + k], omega[k * d], omega_barrett[k * d]);
                    uint64_t t1 = Z::MulFastConst(b[j + l0 + k + 1], omega[(k + 1) * d], omega_barrett[(k + 1) * d]);
                    uint64_t t2 = Z::MulFastConst(b[j + l0 + k + 2], omega[(k + 2) * d], omega_barrett[(k + 2) * d]);
                    uint64_t t3 = Z::MulFastConst(b[j + l0 + k + 3], omega[(k + 3) * d], omega_barrett[(k + 3) * d]);

                    __m256i tV = _mm256_set_epi64x(t3, t2, t1, t0);

                    __m256i bV = _mm256_loadu_si256((__m256i*)&b[j + k]);

                    __m256i addV = _mm256_add_epi64(bV, tV);
                    __m256i add_maskV = _mm256_cmpgt_epi64(addV, p_1V);
                    __m256i add_adjustV = _mm256_sub_epi64(addV, _mm256_and_si256(add_maskV, pV));
                    _mm256_storeu_si256((__m256i*)&b[j + k], add_adjustV);

                    __m256i subV = _mm256_sub_epi64(bV, tV);
                    __m256i sub_maskV = _mm256_cmpgt_epi64(tV, bV);
                    __m256i sub_adjustV = _mm256_add_epi64(subV, _mm256_and_si256(sub_maskV, pV));
                    _mm256_storeu_si256((__m256i*)&b[j + l0 + k], sub_adjustV);
                }
                for (; k < l0; k++) {
                    uint64_t t = Z::MulFastConst(b[j + l0 + k], omega[k * d], omega_barrett[k * d]);
                    b[j + l0 + k] = Z::Sub(b[j + k], t);
                    b[j + k] = Z::Add(b[j + k], t);
                }
            }
        }

        uint64_t z3 = omega[N / 3];
        uint64_t z3_barrett = omega_barrett[N / 3];
        uint64_t zz3 = omega[2 * N / 3];
        uint64_t zz3_barrett = omega_barrett[2 * N / 3];

        for (size_t i = 0; i < v; i++) {
            l0 = U;
            l1 = U * 3;
            for (size_t j = 0; j < i; j++) {
                l0 *= 3;
                l1 *= 3;
            }
            d = N / l1;
            for (size_t j = 0; j < N; j += l1) {
                size_t k = 0;
                for (; k + 2 < l0; k += 4) {
                    uint64_t y01 = Z::MulFastConst(b[j + l0 + k], omega[k * d], omega_barrett[k * d]);
                    uint64_t y02 = Z::MulFastConst(b[j + l0 + l0 + k], omega[2 * k * d], omega_barrett[2 * k * d]);
                    uint64_t y11 = Z::MulFastConst(b[j + l0 + k + 1], omega[(k + 1) * d], omega_barrett[(k + 1) * d]);
                    uint64_t y12 = Z::MulFastConst(b[j + l0 + l0 + k + 1], omega[2 * (k + 1) * d], omega_barrett[2 * (k + 1) * d]);
                    uint64_t y21 = Z::MulFastConst(b[j + l0 + k + 2], omega[(k + 2) * d], omega_barrett[(k + 2) * d]);
                    uint64_t y22 = Z::MulFastConst(b[j + l0 + l0 + k + 2], omega[2 * (k + 2) * d], omega_barrett[2 * (k + 2) * d]);
                    uint64_t y31 = Z::MulFastConst(b[j + l0 + k + 3], omega[(k + 3) * d], omega_barrett[(k + 3) * d]);
                    uint64_t y32 = Z::MulFastConst(b[j + l0 + l0 + k + 3], omega[2 * (k + 3) * d], omega_barrett[2 * (k + 3) * d]);
                    uint64_t y00 = y01 + y02;
                    uint64_t y10 = y11 + y12;
                    uint64_t y20 = y21 + y22;
                    uint64_t y30 = y31 + y32;
                    uint64_t t0 = Z::MulFastConst(y01, z3, z3_barrett) + Z::MulFastConst(y02, zz3, zz3_barrett);
                    uint64_t t1 = Z::MulFastConst(y11, z3, z3_barrett) + Z::MulFastConst(y12, zz3, zz3_barrett);
                    uint64_t t2 = Z::MulFastConst(y21, z3, z3_barrett) + Z::MulFastConst(y22, zz3, zz3_barrett);
                    uint64_t t3 = Z::MulFastConst(y31, z3, z3_barrett) + Z::MulFastConst(y32, zz3, zz3_barrett);
                    
                    __m256i tV = _mm256_set_epi64x(t3, t2, t1, t0);
                    __m256i t_maskV = _mm256_cmpgt_epi64(tV, p_1V);
                    tV = _mm256_sub_epi64(tV, _mm256_and_si256(t_maskV, pV));

                    __m256i yV = _mm256_set_epi64x(y30, y20, y10, y00);
                    __m256i y_maskV = _mm256_cmpgt_epi64(yV, p_1V);
                    yV = _mm256_sub_epi64(yV, _mm256_and_si256(y_maskV, pV));

                    __m256i ytV = _mm256_add_epi64(yV, tV);
                    __m256i yt_maskV = _mm256_cmpgt_epi64(ytV, p_1V);
                    ytV = _mm256_sub_epi64(ytV, _mm256_and_si256(yt_maskV, pV));

                    __m256i bV = _mm256_loadu_si256((__m256i*)&b[j + k]);

                    __m256i addV = _mm256_add_epi64(bV, tV);
                    __m256i add_maskV = _mm256_cmpgt_epi64(addV, p_1V);
                    __m256i add_adjustV = _mm256_sub_epi64(addV, _mm256_and_si256(add_maskV, pV));
                    _mm256_storeu_si256((__m256i*)&b[j + k + l0], add_adjustV);

                    __m256i subV = _mm256_sub_epi64(bV, ytV);
                    __m256i sub_maskV = _mm256_cmpgt_epi64(ytV, bV);
                    __m256i sub_adjustV = _mm256_add_epi64(subV, _mm256_and_si256(sub_maskV, pV));
                    _mm256_storeu_si256((__m256i*)&b[j + k + l0 + l0], sub_adjustV);

                    __m256i b0V = _mm256_add_epi64(bV, yV);
                    __m256i b0_maskV = _mm256_cmpgt_epi64(b0V, p_1V);
                    __m256i b0_adjustV = _mm256_sub_epi64(b0V, _mm256_and_si256(b0_maskV, pV));
                    _mm256_storeu_si256((__m256i*)&b[j + k], b0_adjustV);
                }
                for (; k < l0; k++) {
                    uint64_t y1 = Z::MulFastConst(b[j + l0 + k], omega[k * d], omega_barrett[k * d]);
                    uint64_t y2 = Z::MulFastConst(b[j + l0 + l0 + k], omega[2 * k * d], omega_barrett[2 * k * d]);
                    uint64_t y0 = Z::Add(y1, y2);
                    uint64_t t = Z::Add(Z::MulFastConst(y1, z3, z3_barrett), Z::MulFastConst(y2, zz3, zz3_barrett));
                    b[j + l0 + k] = Z::Add(b[j + k], t);
                    b[j + l0 + l0 + k] = Z::Sub(b[j + k], Z::Add(y0, t));
                    b[j + k] = Z::Add(b[j + k], y0);
                }
            }
        }
        std::copy(b, b + N, a);
    }

    void ForwardCT23NTT(uint64_t a[], uint64_t b[]) {
        CT23NTT(a, b, omega_N_table, omega_N_barrett_table);
    }

    void InverseCT23NTT(uint64_t a[], uint64_t b[]) {
        CT23NTT(a, b, omega_N_inv_table, omega_N_inv_barrett_table);
    }

    static uint64_t omega_O_table[N];
    static uint64_t omega_O_inv_table[N];
    static uint64_t omega_O_barrett_table[N];
    static uint64_t omega_O_inv_barrett_table[N];

    void ComputeOmegaOTable() {
        uint64_t t = omega_O;
        for (size_t i = 1; i <= N; i++) {
            omega_O_barrett_table[(N - gi_inv[i]) % N] = t;
            t = Z::Mul(t, omega_O);
        }
        ForwardCT23NTT(omega_O_barrett_table, omega_O_table);

        uint64_t N_inv = Z::Pow(N, p - 2);
        for (size_t i = 0; i < N; i++) {
            omega_O_inv_table[i] = Z::Pow(omega_O_table[i], p - 2);
            omega_O_table[i] = Z::Mul(omega_O_table[i], N_inv);
            omega_O_inv_table[i] = Z::Mul(omega_O_inv_table[i], N_inv);
            omega_O_barrett_table[i] = Z::ComputeBarrettFactor(omega_O_table[i]);
            omega_O_inv_barrett_table[i] = Z::ComputeBarrettFactor(omega_O_inv_table[i]);
        }
    }

    static bool initialized;

    void ForwardNTT(uint64_t a[]) {
        static uint64_t reg[N];

        for (size_t i = 0; i < N; i++) {
            reg[i] = a[gi[i] - 1];
        }

        ForwardCT23NTT(reg, a);

        for (size_t i = 0; i < N; i++) {
            a[i] = Z::MulFastConst(a[i], omega_O_table[i], omega_O_barrett_table[i]);
        }

        InverseCT23NTT(a, reg);

        for (size_t i = 0; i < N; i++) {
            a[i] = reg[(N - gi_inv[i + 1]) % N];
        }
    }

    void InverseNTT(uint64_t a[]) {
        static uint64_t reg[N];

        for (size_t i = 0; i < N; i++) {
            reg[i] = a[gi[(N - i) % N] - 1];
        }

        ForwardCT23NTT(reg, a);

        for (size_t i = 0; i < N; i++) {
            a[i] = Z::MulFastConst(a[i], omega_O_inv_table[i], omega_O_inv_barrett_table[i]);
        }

        InverseCT23NTT(a, reg);

        for (size_t i = 0; i < N; i++) {
            a[i] = reg[gi_inv[i + 1]];
        }
    }

    static NTT &GetInstance() {
        static NTT instance;
        return instance;
    }

private:
    NTT() {
        ComputeOmegaNTable();
        ComputeOmegaOTable();
    }
};

template <uint64_t p, uint64_t g, size_t O, size_t w>
uint64_t NTT<p, g, O, w>::omega_O_table[N];

template <uint64_t p, uint64_t g, size_t O, size_t w>
uint64_t NTT<p, g, O, w>::omega_O_inv_table[N];

template <uint64_t p, uint64_t g, size_t O, size_t w>
uint64_t NTT<p, g, O, w>::omega_O_barrett_table[N];

template <uint64_t p, uint64_t g, size_t O, size_t w>
uint64_t NTT<p, g, O, w>::omega_O_inv_barrett_table[N];

template <uint64_t p, uint64_t g, size_t O, size_t w>
uint64_t NTT<p, g, O, w>::omega_N_table[N];

template <uint64_t p, uint64_t g, size_t O, size_t w>
uint64_t NTT<p, g, O, w>::omega_N_inv_table[N];

template <uint64_t p, uint64_t g, size_t O, size_t w>
uint64_t NTT<p, g, O, w>::omega_N_barrett_table[N];

template <uint64_t p, uint64_t g, size_t O, size_t w>
uint64_t NTT<p, g, O, w>::omega_N_inv_barrett_table[N];

template <uint64_t p_, uint64_t g_, size_t O_, size_t w_>
class CircNTT {
public:
    using PrimitiveNTT = NTT<p_, g_, O_, w_>;

    constexpr static uint64_t p = p_;
    constexpr static uint64_t g = g_;
    constexpr static size_t O = O_;
    constexpr static size_t w = w_;
    constexpr static size_t N = O;

    using Z = Zp<p>;

    constexpr static uint64_t N_inv = Z::Pow(N, p - 2);

    void ForwardNTT(uint64_t a[]) {
        auto t = a[0];
        for (size_t i = 1; i < N; i++) {
            a[0] = Z::Add(a[0], a[i]);
        }
        PrimitiveNTT::GetInstance().ForwardNTT(a + 1);
        for (size_t i = 1; i < N; i++) {
            a[i] = Z::Add(a[i], t);
        }
    }

    void InverseNTT(uint64_t a[]) {
        PrimitiveNTT::GetInstance().InverseNTT(a + 1);
        auto t = a[0];
        for (size_t i = 1; i < N; i++) {
            t = Z::Sub(t, a[i]);
        }
        a[0] = Z::Mul(t, N_inv);
        for (size_t i = 1; i < N; i++) {
            a[i] = Z::Add(a[i], a[0]);
        }
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