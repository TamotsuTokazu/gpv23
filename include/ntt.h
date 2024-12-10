#ifndef NTT_H
#define NTT_H

#include <cstdint>
#include <cstddef>
#include <array>
#include <iostream>

#include <immintrin.h>
#include <hexl/hexl.hpp>

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

    alignas(64) uint64_t omega_N_table[N];
    alignas(64) uint64_t omega_N_inv_table[N];

    void ComputeOmegaNTable() {
        omega_N_table[0] = 1;
        for (size_t i = 1; i < N; i++) {
            omega_N_table[i] = Z::Mul(omega_N_table[i - 1], omega_N);
        }
        omega_N_inv_table[0] = 1;
        for (size_t i = 1; i < N; i++) {
            omega_N_inv_table[i] = omega_N_table[N - i];
        }
    }

    inline void CT23NTT(uint64_t a[], uint64_t b[], uint64_t omega[]) {

        const uint64_t z3 = omega[N / 3];
        const uint64_t zz3 = omega[2 * N / 3];
        const __m512i z3V =  _mm512_set1_epi64(z3);
        const __m512i zz3V = _mm512_set1_epi64(zz3);

        const __m512i z3zz3V = _mm512_set_epi64(zz3, zz3, zz3, zz3, z3, z3, z3, z3);

        const __m256i pV = _mm256_set1_epi64x(p);
        const __m512i pV512 = _mm512_set1_epi64(p);

        for (size_t i = 0; i < N; i++) {
            b[i] = a[bit_reverse_table[i]];
        }

        for (size_t i = 0; i < N; i += 2) {
            uint64_t t = b[i + 1];
            b[i + 1] = Z::Sub(b[i], t);
            b[i] = Z::Add(b[i], t);
        }

        size_t l0 = 1, l1 = 2, d = N / 2;

        if constexpr (N % 4 == 0) {
            uint64_t ii = omega[N / 4];
            uint64_t ii_barrett = Z::ComputeBarrettFactor(ii);
            for (size_t i = 0; i < N; i += 4) {
                uint64_t t0 = b[i + 2];
                uint64_t t1 = Z::MulFastConst(b[i + 3], ii, ii_barrett);
                b[i + 2] = Z::Sub(b[i], t0);
                b[i + 3] = Z::Sub(b[i + 1], t1);
                b[i] = Z::Add(b[i], t0);
                b[i + 1] = Z::Add(b[i + 1], t1);
            }
        }

        if constexpr (N % 8 == 0) {
            __m512i jV = _mm512_set_epi64(omega[(N / 8) * 7], omega[(N / 8) * 6], omega[(N / 8) * 5], omega[(N / 8) * 4], omega[(N / 8) * 3], omega[(N / 8) * 2], omega[(N / 8) * 1], omega[(N / 8) * 0]);
            for (size_t i = 0; i < N; i += 8) {
                __m512i bV = _mm512_loadu_si512((__m512i*)&b[i]);
                __m512i b0V = _mm512_shuffle_i64x2(bV, bV, 0x44);
                __m512i b1V = _mm512_shuffle_i64x2(bV, bV, 0xEE);
                __m512i tV = Z::Mul512(b1V, jV);
                __m512i rV = _mm512_add_epi64(b0V, tV);
                rV = _mm512_min_epu64(rV, _mm512_sub_epi64(rV, pV512));
                _mm512_storeu_si512((__m512i*)&b[i], rV);
            }

            for (size_t i = 3; i < u; i++) {
                l0 = 1 << i;
                l1 = 1 << (i + 1);
                d = N >> (i + 1);

                for (size_t k = 0; k + 7 < l0; k += 8) {
                    __m512i omegaV = _mm512_set_epi64(omega[(k + 7) * d], omega[(k + 6) * d], omega[(k + 5) * d], omega[(k + 4) * d], omega[(k + 3) * d], omega[(k + 2) * d], omega[(k + 1) * d], omega[(k + 0) * d]);

                    for (size_t j = 0; j < N; j += l1) {
                        // Process 8 elements at a time using AVX-512
                        __m512i bl0V = _mm512_load_si512((__m512i*)&b[j + k + l0]);
                        __m512i tV = Z::Mul512(bl0V, omegaV);
                        __m512i bV = _mm512_load_si512((__m512i*)&b[j + k]);

                        // add path
                        __m512i addV = _mm512_add_epi64(bV, tV);
                        addV = _mm512_min_epu64(addV, _mm512_sub_epi64(addV, pV512));
                        _mm512_store_si512((__m512i*)&b[j + k], addV);

                        // subtract path
                        __m512i subV = _mm512_sub_epi64(bV, tV);
                        // If sub_mask is set, add p to that element
                        subV = _mm512_min_epu64(subV, _mm512_add_epi64(subV, pV512));
                        _mm512_store_si512((__m512i*)&b[j + l0 + k], subV);
                    }
                } // end of AVX-512 block
            }
        }

        for (size_t i = 0; i < v; i++) {
            l0 = U;
            l1 = U * 3;
            for (size_t j = 0; j < i; j++) {
                l0 *= 3;
                l1 *= 3;
            }
            d = N / l1;

            if constexpr (N % 8 == 0) {
                for (size_t k = 0; k + 7 < l0; k += 8) {
                    __m512i omegaV = _mm512_set_epi64(omega[(k + 7) * d], omega[(k + 6) * d], omega[(k + 5) * d], omega[(k + 4) * d], omega[(k + 3) * d], omega[(k + 2) * d], omega[(k + 1) * d], omega[(k + 0) * d]);
                    __m512i omega2V = _mm512_set_epi64(omega[2 * (k + 7) * d], omega[2 * (k + 6) * d], omega[2 * (k + 5) * d], omega[2 * (k + 4) * d], omega[2 * (k + 3) * d], omega[2 * (k + 2) * d], omega[2 * (k + 1) * d], omega[2 * (k + 0) * d]);

                    for (size_t j = 0; j < N; j += l1) {
                        __m512i blV = _mm512_load_si512((__m512i*)&b[j + k + l0]);
                        __m512i y1V = Z::Mul512(blV, omegaV);
                        __m512i bllV = _mm512_load_si512((__m512i*)&b[j + k + l0 + l0]);
                        __m512i y2V = Z::Mul512(bllV, omega2V);
                        __m512i yV = _mm512_add_epi64(y1V, y2V);

                        __m512i yz1V = Z::Mul512(y1V, z3V);
                        __m512i yz2V = Z::Mul512(y2V, zz3V);
                        __m512i tV = _mm512_add_epi64(yz1V, yz2V);
                        
                        tV = _mm512_min_epu64(tV, _mm512_sub_epi64(tV, pV512));
                        yV = _mm512_min_epu64(yV, _mm512_sub_epi64(yV, pV512));

                        __m512i ytV = _mm512_add_epi64(yV, tV);
                        ytV = _mm512_min_epu64(ytV, _mm512_sub_epi64(ytV, pV512));

                        __m512i bV = _mm512_load_si512((__m512i*)&b[j + k]);

                        __m512i addV = _mm512_add_epi64(bV, tV);
                        addV = _mm512_min_epu64(addV, _mm512_sub_epi64(addV, pV512));
                        _mm512_store_si512((__m512i*)&b[j + k + l0], addV);

                        __m512i subV = _mm512_sub_epi64(bV, ytV);
                        subV = _mm512_min_epu64(subV, _mm512_add_epi64(subV, pV512));
                        _mm512_store_si512((__m512i*)&b[j + k + l0 + l0], subV);

                        __m512i b0V = _mm512_add_epi64(bV, yV);
                        b0V = _mm512_min_epu64(b0V, _mm512_sub_epi64(b0V, pV512));
                        _mm512_store_si512((__m512i*)&b[j + k], b0V);
                    }
                } // end of AVX-512 block

            } else if constexpr (N % 4 == 0) {
                for (size_t k = 0; k + 3 < l0; k += 4) {
                    __m512i omegaV = _mm512_set_epi64(omega[2 * (k + 3) * d], omega[2 * (k + 2) * d], omega[2 * (k + 1) * d], omega[2 * (k + 0) * d], omega[(k + 3) * d], omega[(k + 2) * d], omega[(k + 1) * d], omega[(k + 0) * d]);

                    for (size_t j = 0; j < N; j += l1) {

                        __m512i bl0V = _mm512_loadu_si512((__m512*)&b[j + l0 + k]);
                        __m512i bl1V = _mm512_loadu_si512((__m512*)&b[j + l0 + l0 + k - 4]);
                        __m512i blV = _mm512_shuffle_i64x2(bl0V, bl1V, 0xE4);

                        __m512i ypreV = Z::Mul512(blV, omegaV);
                        __m256i yV = _mm256_add_epi64(_mm512_extracti64x4_epi64(ypreV, 0), _mm512_extracti64x4_epi64(ypreV, 1));

                        __m512i tpreV = Z::Mul512(ypreV, z3zz3V);
                        __m256i tV = _mm256_add_epi64(_mm512_extracti64x4_epi64(tpreV, 0), _mm512_extracti64x4_epi64(tpreV, 1));

                        yV = _mm256_min_epu64(yV, _mm256_sub_epi64(yV, pV));
                        tV = _mm256_min_epu64(tV, _mm256_sub_epi64(tV, pV));

                        __m256i ytV = _mm256_add_epi64(yV, tV);
                        ytV = _mm256_min_epu64(ytV, _mm256_sub_epi64(ytV, pV));

                        __m256i bV = _mm256_loadu_si256((__m256i*)&b[j + k]);

                        __m256i addV = _mm256_add_epi64(bV, tV);
                        addV = _mm256_min_epu64(addV, _mm256_sub_epi64(addV, pV));
                        _mm256_storeu_si256((__m256i*)&b[j + k + l0], addV);

                        __m256i subV = _mm256_sub_epi64(bV, ytV);
                        subV = _mm256_min_epu64(subV, _mm256_add_epi64(subV, pV));
                        _mm256_storeu_si256((__m256i*)&b[j + k + l0 + l0], subV);

                        __m256i b0V = _mm256_add_epi64(bV, yV);
                        b0V = _mm256_min_epu64(b0V, _mm256_sub_epi64(b0V, pV));
                        _mm256_storeu_si256((__m256i*)&b[j + k], b0V);
                    }
                } // end of AVX-256 block

            } else {
                for (size_t j = 0; j < N; j += l1) {
                    for (size_t k = 0; k < l0; k++) {
                        uint64_t y1 = Z::Mul(b[j + l0 + k], omega[k * d]);
                        uint64_t y2 = Z::Mul(b[j + l0 + l0 + k], omega[2 * k * d]);
                        uint64_t y0 = Z::Add(y1, y2);
                        uint64_t t = Z::Add(Z::Mul(y1, z3), Z::Mul(y2, zz3));
                        b[j + l0 + k] = Z::Add(b[j + k], t);
                        b[j + l0 + l0 + k] = Z::Sub(b[j + k], Z::Add(y0, t));
                        b[j + k] = Z::Add(b[j + k], y0);
                    }
                }
            }
        }
        std::copy(b, b + N, a);
    }

    void ForwardCT23NTT(uint64_t a[], uint64_t b[]) {
        CT23NTT(a, b, omega_N_table);
    }

    void InverseCT23NTT(uint64_t a[], uint64_t b[]) {
        CT23NTT(a, b, omega_N_inv_table);
    }

    alignas(64) uint64_t omega_O_table[N];
    alignas(64) uint64_t omega_O_inv_table[N];

    void ComputeOmegaOTable() {
        uint64_t t = omega_O;
        for (size_t i = 1; i <= N; i++) {
            omega_O_inv_table[(N - gi_inv[i]) % N] = t;
            t = Z::Mul(t, omega_O);
        }
        ForwardCT23NTT(omega_O_inv_table, omega_O_table);

        uint64_t N_inv = Z::Pow(N, p - 2);
        for (size_t i = 0; i < N; i++) {
            omega_O_inv_table[i] = Z::Pow(omega_O_table[i], p - 2);
            omega_O_table[i] = Z::Mul(omega_O_table[i], N_inv);
            omega_O_inv_table[i] = Z::Mul(omega_O_inv_table[i], N_inv);
        }
    }

    void ForwardNTT(uint64_t a[]) {
        alignas(64) static uint64_t reg[N];
        alignas(64) static uint64_t reg2[N];

        for (size_t i = 0; i < N; i++) {
            reg[i] = a[gi[i] - 1];
        }

        ForwardCT23NTT(reg, reg2);

        if constexpr (N % 8 == 0) {

            for (size_t i = 0; i < N; i += 8) {
                __m512i reg2V = _mm512_load_si512((__m512i*)&reg2[i]);
                __m512i omega_O_tableV = _mm512_load_si512((__m512i*)&omega_O_table[i]);
                __m512i bV = Z::Mul512(reg2V, omega_O_tableV);
                _mm512_store_si512((__m512i*)&reg2[i], bV);
            }

        } else {

            intel::hexl::EltwiseMultMod(reg2, reg2, omega_O_table, N, p, 1);

        }

        InverseCT23NTT(reg2, reg);

        for (size_t i = 0; i < N; i++) {
            a[i] = reg[(N - gi_inv[i + 1]) % N];
        }
    }

    void InverseNTT(uint64_t a[]) {
        alignas(64) static uint64_t reg[N];
        alignas(64) static uint64_t reg2[N];

        for (size_t i = 0; i < N; i++) {
            reg[i] = a[gi[(N - i) % N] - 1];
        }

        ForwardCT23NTT(reg, reg2);

        intel::hexl::EltwiseMultMod(reg2, reg2, omega_O_inv_table, N, p, 1);

        InverseCT23NTT(reg2, reg);

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

template <uint64_t p_, uint64_t g_, size_t O_, size_t w_>
class NTT23 {

public:

    constexpr static uint64_t p = p_;
    constexpr static uint64_t g = g_;
    constexpr static size_t O = O_;
    constexpr static size_t w = w_;
    constexpr static size_t N = O - 1;

    using Z = Zp<p>;

    static_assert(p % ((u_int64_t)O * (O - 1)) == 1, "p must be 1 mod O * (O - 1)");

    constexpr static size_t u = []() {
        size_t u = 0;
        size_t N = O - 1;
        while (N % 2 == 0) {
            N >>= 1;
            u++;
        }
        return u;
    }();
    constexpr static size_t U = 1 << u;

    static_assert(3 * U == N, "O should be 3 * 2^u + 1");

    constexpr static uint64_t omega = Z::Pow(g, (p - 1) / O);
    constexpr static uint64_t omega_inv = Z::Inv(omega);

    constexpr static uint64_t alpha = Z::Pow(g, (p - 1) / U);
    constexpr static uint64_t alpha_inv = Z::Inv(alpha);

    constexpr static uint64_t beta = Z::Pow(g, (p - 1) / 3);
    constexpr static uint64_t beta_inv = Z::Inv(beta);

    constexpr static uint64_t nu = Z::Pow(g, (p - 1) / N);
    constexpr static uint64_t nu_inv = Z::Inv(nu);

    constexpr static std::array<size_t, U> bit_reverse_table = []() {
        std::array<size_t, U> bit_reverse_table;
        for (size_t i = 0, j = 0; i < U; i++) {
            bit_reverse_table[i] = j;
            for (size_t k = U >> 1; (j ^= k) < k; k >>= 1);
        }
        return bit_reverse_table;
    }();

    template <uint64_t alpha_>
    inline void Base2NTT(uint64_t b[]) {
        constexpr static std::array<uint64_t, U> alpha_table = []() {
            std::array<uint64_t, U> alpha_table;
            alpha_table[0] = 1;
            for (size_t i = 1; i < U; i++) {
                alpha_table[i] = Z::Mul(alpha_table[i - 1], alpha_);
            }
            return alpha_table;
        }();

        const __m512i pV = _mm512_set1_epi64(p);

        for (size_t i = 0; i < U; i += 2) {
            uint64_t t = b[i + 1];
            b[i + 1] = Z::Sub(b[i], t);
            b[i] = Z::Add(b[i], t);
        }

        if constexpr (U % 4 == 0) {
            uint64_t ii = alpha_table[U / 4];
            uint64_t ii_barrett = Z::ComputeBarrettFactor(ii);
            for (size_t i = 0; i < U; i += 4) {
                uint64_t t0 = b[i + 2];
                uint64_t t1 = Z::MulFastConst(b[i + 3], ii, ii_barrett);
                b[i + 2] = Z::Sub(b[i], t0);
                b[i + 3] = Z::Sub(b[i + 1], t1);
                b[i] = Z::Add(b[i], t0);
                b[i + 1] = Z::Add(b[i + 1], t1);
            }
        }

        if constexpr (U % 8 == 0) {
            __m512i jV = _mm512_set_epi64(alpha_table[(U / 8) * 7], alpha_table[(U / 8) * 6], alpha_table[(U / 8) * 5], alpha_table[(U / 8) * 4], alpha_table[(U / 8) * 3], alpha_table[(U / 8) * 2], alpha_table[(U / 8) * 1], alpha_table[(U / 8) * 0]);
            for (size_t i = 0; i < U; i += 8) {
                __m512i bV = _mm512_loadu_si512((__m512i*)&b[i]);
                __m512i b0V = _mm512_shuffle_i64x2(bV, bV, 0x44);
                __m512i b1V = _mm512_shuffle_i64x2(bV, bV, 0xEE);
                __m512i tV = Z::Mul512(b1V, jV);
                __m512i rV = _mm512_add_epi64(b0V, tV);
                rV = _mm512_min_epu64(rV, _mm512_sub_epi64(rV, pV));
                _mm512_storeu_si512((__m512i*)&b[i], rV);
            }

            for (size_t i = 3; i < u; i++) {
                size_t l0 = 1 << i;
                size_t l1 = 1 << (i + 1);
                size_t d = U >> (i + 1);

                for (size_t k = 0; k + 7 < l0; k += 8) {
                    __m512i omegaV = _mm512_set_epi64(alpha_table[(k + 7) * d], alpha_table[(k + 6) * d], alpha_table[(k + 5) * d], alpha_table[(k + 4) * d], alpha_table[(k + 3) * d], alpha_table[(k + 2) * d], alpha_table[(k + 1) * d], alpha_table[(k + 0) * d]);

                    for (size_t j = 0; j < U; j += l1) {
                        // Process 8 elements at a time using AVX-512
                        __m512i bl0V = _mm512_loadu_si512((__m512i*)&b[j + k + l0]);
                        __m512i tV = Z::Mul512(bl0V, omegaV);
                        __m512i bV = _mm512_loadu_si512((__m512i*)&b[j + k]);

                        // add path
                        __m512i addV = _mm512_add_epi64(bV, tV);
                        addV = _mm512_min_epu64(addV, _mm512_sub_epi64(addV, pV));
                        _mm512_storeu_si512((__m512i*)&b[j + k], addV);

                        // subtract path
                        __m512i subV = _mm512_sub_epi64(bV, tV);
                        // If sub_mask is set, add p to that element
                        subV = _mm512_min_epu64(subV, _mm512_add_epi64(subV, pV));
                        _mm512_storeu_si512((__m512i*)&b[j + l0 + k], subV);
                    }
                } // end of AVX-512 block
            }
        }
    }

    template <uint64_t beta_>
    inline void Base3NTT(uint64_t b[]) {
        __m512i x1V = _mm512_set1_epi64(beta_);
        __m512i x2V = _mm512_set1_epi64(Z::Mul(beta_, beta_));
        __m512i pV = _mm512_set1_epi64(p);
        __m512i p2V = _mm512_set1_epi64(p * 2);

        if constexpr (U % 8 == 0) {
            for (size_t i = 0; i < U; i += 8) {
                __m512i b0V = _mm512_loadu_si512((__m512i*)&b[i]);
                __m512i b1V = _mm512_loadu_si512((__m512i*)&b[i + U]);
                __m512i b2V = _mm512_loadu_si512((__m512i*)&b[i + 2 * U]);

                __m512i y1V = Z::Mul512(b1V, x1V);
                __m512i y2V = Z::Mul512(b2V, x2V);

                __m512i t0V = _mm512_add_epi64(b1V, b2V);
                t0V = _mm512_min_epu64(t0V, _mm512_sub_epi64(t0V, pV));

                __m512i t1V = _mm512_add_epi64(y1V, y2V);
                t1V = _mm512_min_epu64(t1V, _mm512_sub_epi64(t1V, pV));

                __m512i t2V = _mm512_sub_epi64(p2V, _mm512_add_epi64(t0V, t1V));

                __m512i r0V = _mm512_add_epi64(b0V, t0V);
                r0V = _mm512_min_epu64(r0V, _mm512_sub_epi64(r0V, pV));

                __m512i r1V = _mm512_add_epi64(b0V, t1V);
                r1V = _mm512_min_epu64(r1V, _mm512_sub_epi64(r1V, pV));

                __m512i r2V = _mm512_add_epi64(b0V, t2V);
                r2V = _mm512_min_epu64(r2V, _mm512_sub_epi64(r2V, p2V));
                r2V = _mm512_min_epu64(r2V, _mm512_sub_epi64(r2V, pV));

                _mm512_storeu_si512((__m512i*)&b[i], r0V);
                _mm512_storeu_si512((__m512i*)&b[i + U], r1V);
                _mm512_storeu_si512((__m512i*)&b[i + 2 * U], r2V);
            }
        } else {
            for (size_t i = 0; i < U; i++) {
                uint64_t y1 = Z::Mul(b[i + U], beta_);
                uint64_t y2 = Z::Mul(b[i + 2 * U], Z::Mul(beta_, beta_));
                uint64_t t0 = Z::Add(b[i + U], b[i + 2 * U]);
                uint64_t t1 = Z::Add(y1, y2);
                uint64_t t2 = Z::Sub(p * 2, Z::Add(y1, y2));
                b[i] = Z::Add(b[i], t0);
                b[i + U] = Z::Add(b[i], t1);
                b[i + 2 * U] = Z::Add(b[i], t2);
            }
        }
    }

    inline void ForwardTensor23NTT(uint64_t a[]) {
        Base2NTT<alpha>(a);
        Base2NTT<alpha>(a + U);
        Base2NTT<alpha>(a + 2 * U);
        Base3NTT<beta>(a);
    }

    inline void InverseTensor23NTT(uint64_t a[]) {
        Base3NTT<beta_inv>(a);
        Base2NTT<alpha_inv>(a);
        Base2NTT<alpha_inv>(a + U);
        Base2NTT<alpha_inv>(a + 2 * U);
    }

    constexpr static std::array<size_t, N> gi = []() {
        std::array<size_t, N> gi;
        gi[0] = 1;
        for (size_t i = 1; i < N; i++) {
            gi[i] = (uint64_t)gi[i - 1] * w % O;
        }
        return gi;
    }();

    constexpr static std::array<size_t, O> gi_inv = []() {
        std::array<size_t, O> gi_inv;
        gi_inv[0] = -1;
        for (size_t i = 0; i < N; i++) {
            gi_inv[gi[i]] = i;
        }
        return gi_inv;
    }();

    constexpr static std::array<size_t, N> t0_perm = []() {
        std::array<size_t, N> t0_perm;
        for (size_t i = 0; i < U; i++) {
            for (size_t j = 0; j < 3; j++) {
                t0_perm[j * U + i] = (i * 3 + j * U) % N;
            }
        }
        return t0_perm;
    }();

    constexpr static std::array<size_t, N> t0_perm_inv = []() {
        std::array<size_t, N> t0_perm_inv;
        for (size_t i = 0; i < N; i++) {
            t0_perm_inv[t0_perm[i]] = i;
        }
        return t0_perm_inv;
    }();

    alignas(64) uint64_t omega_table[N];
    alignas(64) uint64_t omega_inv_table[N];

    void ComputeOmegaOTable() {
        uint64_t t = omega;
        for (size_t i = 1; i <= N; i++) {
            omega_inv_table[(N - gi_inv[i]) % N] = t;
            t = Z::Mul(t, omega);
        }

        for (size_t i = 0; i < N; i++) {
            omega_table[i] = omega_inv_table[t0_perm[i]];
        }

        ForwardTensor23NTT(omega_table);
        uint64_t N_inv = Z::Pow(N, p - 2);
        for (size_t i = 0; i < N; i++) {
            omega_inv_table[i] = Z::Inv(omega_table[i]);
            omega_table[i] = Z::Mul(omega_table[i], N_inv);
            omega_inv_table[i] = Z::Mul(omega_inv_table[i], N_inv);
        }
    }

    inline void ForwardNTT(uint64_t a[]) {

        constexpr static std::array<size_t, N> pre_perm = []() {
            std::array<size_t, N> pre_perm;
            for (size_t i = 0; i < N; i++) {
                pre_perm[i] = gi[t0_perm[i]] - 1;
            }
            return pre_perm;
        }();

        constexpr static std::array<size_t, N> post_perm = []() {
            std::array<size_t, N> post_perm;
            for (size_t i = 0; i < N; i++) {
                post_perm[i] = t0_perm_inv[(N - gi_inv[i + 1]) % N];
            }
            return post_perm;
        }();

        alignas(64) static uint64_t reg[N];
        alignas(64) static uint64_t reg2[N];
        
        for (size_t i = 0; i < N; i++) {
            reg[i] = a[pre_perm[i]];
        }

        for (size_t i = 0; i < N; i++) {
            reg2[i] = reg[bit_reverse_table[i]];
        }

        ForwardTensor23NTT(reg2);

        for (size_t i = 0; i < N; i++) {
            reg[i] = Z::Mul(reg2[i], omega_table[i]);
        }

        for (size_t i = 0; i < N; i++) {
            reg2[i] = reg[bit_reverse_table[i]];
        }

        InverseTensor23NTT(reg2);

        for (size_t i = 0; i < N; i++) {
            a[i] = reg2[post_perm[i]];
        }
    }

    inline void InverseNTT(uint64_t a[]) {

        constexpr static std::array<size_t, N> pre_perm = []() {
            std::array<size_t, N> pre_perm;
            for (size_t i = 0; i < N; i++) {
                pre_perm[i] = gi[(N - t0_perm[i]) % N] - 1;
            }
            return pre_perm;
        }();

        constexpr static std::array<size_t, N> post_perm = []() {
            std::array<size_t, N> post_perm;
            for (size_t i = 0; i < N; i++) {
                post_perm[i] = t0_perm_inv[gi_inv[i + 1]];
            }
            return post_perm;
        }();

        alignas(64) static uint64_t reg[N];
        alignas(64) static uint64_t reg2[N];

        for (size_t i = 0; i < N; i++) {
            reg[i] = a[pre_perm[i]];
        }

        for (size_t i = 0; i < N; i++) {
            reg2[i] = reg[bit_reverse_table[i]];
        }

        ForwardTensor23NTT(reg2);

        for (size_t i = 0; i < N; i++) {
            reg[i] = Z::Mul(reg2[i], omega_inv_table[i]);
        }

        InverseTensor23NTT(reg2);

        for (size_t i = 0; i < N; i++) {
            a[i] = reg2[post_perm[i]];
        }
    }

    static inline NTT23 &GetInstance() {
        static NTT23 instance;
        return instance;
    }

private:

    NTT23() {
        ComputeOmegaOTable();
    }

};

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