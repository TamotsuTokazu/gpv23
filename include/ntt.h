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
    static_assert(u >= 5, "u should be at least 5");

    constexpr static uint64_t omega = Z::Pow(g, (p - 1) / O);
    constexpr static uint64_t omega_inv = Z::Inv(omega);

    constexpr static uint64_t alpha = Z::Pow(g, (p - 1) / U);
    constexpr static uint64_t alpha_inv = Z::Inv(alpha);

    constexpr static uint64_t beta = Z::Pow(g, (p - 1) / 3);
    constexpr static uint64_t beta_inv = Z::Inv(beta);

    constexpr static uint64_t nu = Z::Pow(g, (p - 1) / N);
    constexpr static uint64_t nu_inv = Z::Inv(nu);

    alignas(64) constexpr static std::array<size_t, N> bit_reverse_table = []() {
        std::array<size_t, N> bit_reverse_table;
        for (size_t i = 0, j = 0; i < U; i++) {
            bit_reverse_table[i] = j;
            for (size_t k = U >> 1; (j ^= k) < k; k >>= 1);
        }
        for (size_t i = U; i < N; i++) {
            bit_reverse_table[i] = bit_reverse_table[i - U] + U;
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
        const __m512i p2V = _mm512_set1_epi64(p * 2);

        for (size_t i = 0; i < U; i += 2) {
            uint64_t t = b[i + 1];
            b[i + 1] = b[i] + p - t;
            b[i] = b[i] + t;
        }

        const uint64_t ii = alpha_table[U / 4];
        const __m512i iiV = _mm512_set1_epi64(ii);
        const __m512i idx0V = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        const __m512i idx1V = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        const __m512i jdx0V = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
        const __m512i jdx1V = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
        for (size_t i = 0; i < U; i += 32) {

            __m512i a0V = _mm512_loadu_si512((__m512i*)&b[i]);      //  0  1  2  3  4  5  6  7
            __m512i a1V = _mm512_loadu_si512((__m512i*)&b[i + 8]);  //  8  9 10 11 12 13 14 15
            __m512i a2V = _mm512_loadu_si512((__m512i*)&b[i + 16]); // 16 17 18 19 20 21 22 23
            __m512i a3V = _mm512_loadu_si512((__m512i*)&b[i + 24]); // 24 25 26 27 28 29 30 31

            __m512i x0V = _mm512_permutex2var_epi64(a0V, idx0V, a1V); //  0  2  4  6  8 10 12 14
            __m512i x1V = _mm512_permutex2var_epi64(a0V, idx1V, a1V); //  1  3  5  7  9 11 13 15
            __m512i x2V = _mm512_permutex2var_epi64(a2V, idx0V, a3V); // 16 18 20 22 24 26 28 30
            __m512i x3V = _mm512_permutex2var_epi64(a2V, idx1V, a3V); // 17 19 21 23 25 27 29 31

            __m512i b0V = _mm512_permutex2var_epi64(x0V, idx0V, x2V); //  0  4  8 12 16 20 24 28
            __m512i b1V = _mm512_permutex2var_epi64(x1V, idx0V, x3V); //  1  5  9 13 17 21 25 29
            __m512i b2V = _mm512_permutex2var_epi64(x0V, idx1V, x2V); //  2  6 10 14 18 22 26 30
            __m512i b3V = _mm512_permutex2var_epi64(x1V, idx1V, x3V); //  3  7 11 15 19 23 27 31

            b3V = Z::Mul512(b3V, iiV);

            __m512i y0V = _mm512_add_epi64(b0V, b2V);
            __m512i y1V = _mm512_add_epi64(b1V, b3V);
            __m512i y2V = _mm512_add_epi64(b0V, _mm512_sub_epi64(p2V, b2V));
            __m512i y3V = _mm512_add_epi64(b1V, _mm512_sub_epi64(pV, b3V));

            __m512i z0V = _mm512_permutex2var_epi64(y0V, jdx0V, y2V); //  0  2  4  6  8 10 12 14
            __m512i z1V = _mm512_permutex2var_epi64(y0V, jdx1V, y2V); // 16 18 20 22 24 26 28 30
            __m512i z2V = _mm512_permutex2var_epi64(y1V, jdx0V, y3V); //  1  3  5  7  9 11 13 15
            __m512i z3V = _mm512_permutex2var_epi64(y1V, jdx1V, y3V); // 17 19 21 23 25 27 29 31

            __m512i r0V = _mm512_permutex2var_epi64(z0V, jdx0V, z2V); //  0  1  2  3  4  5  6  7
            __m512i r1V = _mm512_permutex2var_epi64(z0V, jdx1V, z2V); //  8  9 10 11 12 13 14 15
            __m512i r2V = _mm512_permutex2var_epi64(z1V, jdx0V, z3V); // 16 17 18 19 20 21 22 23
            __m512i r3V = _mm512_permutex2var_epi64(z1V, jdx1V, z3V); // 24 25 26 27 28 29 30 31

            _mm512_storeu_si512((__m512i*)&b[i + 0], r0V);
            _mm512_storeu_si512((__m512i*)&b[i + 8], r1V);
            _mm512_storeu_si512((__m512i*)&b[i + 16], r2V);
            _mm512_storeu_si512((__m512i*)&b[i + 24], r3V);

        }

        __m512i jV = _mm512_set_epi64(alpha_table[(U / 8) * 3], alpha_table[(U / 8) * 2], alpha_table[(U / 8) * 1], alpha_table[(U / 8) * 0], alpha_table[(U / 8) * 3], alpha_table[(U / 8) * 2], alpha_table[(U / 8) * 1], alpha_table[(U / 8) * 0]);
        for (size_t i = 0; i < U; i += 16) {
            __m512i bV = _mm512_loadu_si512((__m512i*)&b[i]);
            __m512i bbV = _mm512_loadu_si512((__m512i*)&b[i + 8]);
            __m512i b0V = _mm512_shuffle_i64x2(bV, bbV, 0x44);
            __m512i b1V = _mm512_shuffle_i64x2(bV, bbV, 0xEE);
            b0V = _mm512_min_epu64(b0V, _mm512_sub_epi64(b0V, p2V));
            __m512i tV = Z::Mul512(b1V, jV);
            __m512i y0V = _mm512_add_epi64(b0V, tV);
            __m512i y1V = _mm512_add_epi64(b0V, _mm512_sub_epi64(pV, tV));
            __m512i r0V = _mm512_shuffle_i64x2(y0V, y1V, 0x44);
            __m512i r1V = _mm512_shuffle_i64x2(y0V, y1V, 0xEE);
            _mm512_storeu_si512((__m512i*)&b[i], r0V);
            _mm512_storeu_si512((__m512i*)&b[i + 8], r1V);
        }

        for (size_t i = 3; i + 1 < u; i++) {
            size_t l0 = 1 << i;
            size_t l1 = 1 << (i + 1);
            size_t d = U >> (i + 1);

            for (size_t k = 0; k + 7 < l0; k += 8) {
                __m512i omegaV = _mm512_set_epi64(alpha_table[(k + 7) * d], alpha_table[(k + 6) * d], alpha_table[(k + 5) * d], alpha_table[(k + 4) * d], alpha_table[(k + 3) * d], alpha_table[(k + 2) * d], alpha_table[(k + 1) * d], alpha_table[(k + 0) * d]);

                for (size_t j = 0; j < U; j += l1) {
                    __m512i bl0V = _mm512_loadu_si512((__m512i*)&b[j + k + l0]);
                    __m512i tV = Z::Mul512(bl0V, omegaV);
                    __m512i bV = _mm512_loadu_si512((__m512i*)&b[j + k]);

                    bV = _mm512_min_epu64(bV, _mm512_sub_epi64(bV, p2V));

                    __m512i addV = _mm512_add_epi64(bV, tV);
                    _mm512_storeu_si512((__m512i*)&b[j + k], addV);

                    __m512i subV = _mm512_add_epi64(bV, _mm512_sub_epi64(pV, tV));
                    _mm512_storeu_si512((__m512i*)&b[j + l0 + k], subV);
                }
            } // end of AVX-512 block
        }

        size_t l0 = U >> 1;
        size_t l1 = U;

        for (size_t k = 0; k + 7 < l0; k += 8) {
            __m512i omegaV = _mm512_loadu_si512((__m512i*)&alpha_table[k]);

            for (size_t j = 0; j < U; j += l1) {

                __m512i bl0V = _mm512_loadu_si512((__m512i*)&b[j + k + l0]);
                __m512i tV = Z::Mul512(bl0V, omegaV);
                __m512i bV = _mm512_loadu_si512((__m512i*)&b[j + k]);

                __m512i addV = _mm512_add_epi64(bV, tV);
                _mm512_storeu_si512((__m512i*)&b[j + k], addV);

                __m512i subV = _mm512_add_epi64(bV, _mm512_sub_epi64(pV, tV));
                _mm512_storeu_si512((__m512i*)&b[j + l0 + k], subV);

            }
        }
    }

    template <uint64_t alpha_>
    inline void InverseBase2NTT(uint64_t b[]) {
        constexpr static std::array<uint64_t, U> alpha_table = []() {
            std::array<uint64_t, U> alpha_table;
            alpha_table[0] = 1;
            for (size_t i = 1; i < U; i++) {
                alpha_table[i] = Z::Mul(alpha_table[i - 1], alpha_);
            }
            return alpha_table;
        }();

        // const __m512i pV = _mm512_set1_epi64(p);
        const __m512i p2V = _mm512_set1_epi64(p * 2);

        size_t l0 = U >> 1;
        size_t l1 = U;

        for (size_t k = 0; k + 7 < l0; k += 8) {
            __m512i omegaV = _mm512_loadu_si512((__m512i*)&alpha_table[k]);

            for (size_t j = 0; j < U; j += l1) {

                __m512i b0V = _mm512_loadu_si512((__m512i*)&b[j + k]);
                __m512i b1V = _mm512_loadu_si512((__m512i*)&b[j + k + l0]);

                __m512i addV = _mm512_add_epi64(b0V, b1V);
                _mm512_storeu_si512((__m512i*)&b[j + k], addV);

                __m512i subV = _mm512_add_epi64(b0V, _mm512_sub_epi64(p2V, b1V));
                subV = Z::Mul512(subV, omegaV);
                _mm512_storeu_si512((__m512i*)&b[j + l0 + k], subV);

            }
        }

        for (size_t i = u - 2; i >= 3; i--) {
            size_t l0 = 1 << i;
            size_t l1 = 1 << (i + 1);
            size_t d = U >> (i + 1);

            for (size_t k = 0; k + 7 < l0; k += 8) {
                __m512i omegaV = _mm512_set_epi64(alpha_table[(k + 7) * d], alpha_table[(k + 6) * d], alpha_table[(k + 5) * d], alpha_table[(k + 4) * d], alpha_table[(k + 3) * d], alpha_table[(k + 2) * d], alpha_table[(k + 1) * d], alpha_table[(k + 0) * d]);

                for (size_t j = 0; j < U; j += l1) {
                    // Process 8 elements at a time using AVX-512
                    __m512i b0V = _mm512_loadu_si512((__m512i*)&b[j + k]);
                    __m512i b1V = _mm512_loadu_si512((__m512i*)&b[j + k + l0]);

                    // add path
                    __m512i addV = _mm512_add_epi64(b0V, b1V);
                    addV = _mm512_min_epu64(addV, _mm512_sub_epi64(addV, p2V));
                    _mm512_storeu_si512((__m512i*)&b[j + k], addV);

                    // subtract path
                    __m512i subV = _mm512_add_epi64(b0V, _mm512_sub_epi64(p2V, b1V));
                    subV = Z::Mul512(subV, omegaV);
                    _mm512_storeu_si512((__m512i*)&b[j + l0 + k], subV);
                }
            } // end of AVX-512 block
        }

        __m512i jV = _mm512_set_epi64(alpha_table[(U / 8) * 3], alpha_table[(U / 8) * 2], alpha_table[(U / 8) * 1], alpha_table[(U / 8) * 0], alpha_table[(U / 8) * 3], alpha_table[(U / 8) * 2], alpha_table[(U / 8) * 1], alpha_table[(U / 8) * 0]);
        for (size_t i = 0; i < U; i += 16) {
            __m512i bV = _mm512_loadu_si512((__m512i*)&b[i]);
            __m512i bbV = _mm512_loadu_si512((__m512i*)&b[i + 8]);
            __m512i b0V = _mm512_shuffle_i64x2(bV, bbV, 0x44);
            __m512i b1V = _mm512_shuffle_i64x2(bV, bbV, 0xEE);

            __m512i y0V = _mm512_add_epi64(b0V, b1V);
            y0V = _mm512_min_epu64(y0V, _mm512_sub_epi64(y0V, p2V));

            __m512i y1V = _mm512_add_epi64(b0V, _mm512_sub_epi64(p2V, b1V));
            y1V = Z::Mul512(y1V, jV);

            __m512i r0V = _mm512_shuffle_i64x2(y0V, y1V, 0x44);
            __m512i r1V = _mm512_shuffle_i64x2(y0V, y1V, 0xEE);
            _mm512_storeu_si512((__m512i*)&b[i], r0V);
            _mm512_storeu_si512((__m512i*)&b[i + 8], r1V);
        }

        const uint64_t ii = alpha_table[U / 4];
        const __m512i iiV = _mm512_set1_epi64(ii);
        const __m512i idx0V = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        const __m512i idx1V = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        const __m512i jdx0V = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
        const __m512i jdx1V = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
        for (size_t i = 0; i < U; i += 32) {

            __m512i a0V = _mm512_loadu_si512((__m512i*)&b[i]);      //  0  1  2  3  4  5  6  7
            __m512i a1V = _mm512_loadu_si512((__m512i*)&b[i + 8]);  //  8  9 10 11 12 13 14 15
            __m512i a2V = _mm512_loadu_si512((__m512i*)&b[i + 16]); // 16 17 18 19 20 21 22 23
            __m512i a3V = _mm512_loadu_si512((__m512i*)&b[i + 24]); // 24 25 26 27 28 29 30 31

            __m512i x0V = _mm512_permutex2var_epi64(a0V, idx0V, a1V); //  0  2  4  6  8 10 12 14
            __m512i x1V = _mm512_permutex2var_epi64(a0V, idx1V, a1V); //  1  3  5  7  9 11 13 15
            __m512i x2V = _mm512_permutex2var_epi64(a2V, idx0V, a3V); // 16 18 20 22 24 26 28 30
            __m512i x3V = _mm512_permutex2var_epi64(a2V, idx1V, a3V); // 17 19 21 23 25 27 29 31

            __m512i b0V = _mm512_permutex2var_epi64(x0V, idx0V, x2V); //  0  4  8 12 16 20 24 28
            __m512i b1V = _mm512_permutex2var_epi64(x1V, idx0V, x3V); //  1  5  9 13 17 21 25 29
            __m512i b2V = _mm512_permutex2var_epi64(x0V, idx1V, x2V); //  2  6 10 14 18 22 26 30
            __m512i b3V = _mm512_permutex2var_epi64(x1V, idx1V, x3V); //  3  7 11 15 19 23 27 31

            __m512i y0V = _mm512_add_epi64(b0V, b2V);
            __m512i y1V = _mm512_add_epi64(b1V, b3V);
            __m512i y2V = _mm512_add_epi64(b0V, _mm512_sub_epi64(p2V, b2V));
            __m512i y3V = _mm512_add_epi64(b1V, _mm512_sub_epi64(p2V, b3V));

            y0V = _mm512_min_epu64(y0V, _mm512_sub_epi64(y0V, p2V));
            y1V = _mm512_min_epu64(y1V, _mm512_sub_epi64(y1V, p2V));
            y2V = _mm512_min_epu64(y2V, _mm512_sub_epi64(y2V, p2V));
            y3V = Z::Mul512(y3V, iiV);

            __m512i z0V = _mm512_permutex2var_epi64(y0V, jdx0V, y2V); //  0  2  4  6  8 10 12 14
            __m512i z1V = _mm512_permutex2var_epi64(y0V, jdx1V, y2V); // 16 18 20 22 24 26 28 30
            __m512i z2V = _mm512_permutex2var_epi64(y1V, jdx0V, y3V); //  1  3  5  7  9 11 13 15
            __m512i z3V = _mm512_permutex2var_epi64(y1V, jdx1V, y3V); // 17 19 21 23 25 27 29 31

            __m512i r0V = _mm512_permutex2var_epi64(z0V, jdx0V, z2V); //  0  1  2  3  4  5  6  7
            __m512i r1V = _mm512_permutex2var_epi64(z0V, jdx1V, z2V); //  8  9 10 11 12 13 14 15
            __m512i r2V = _mm512_permutex2var_epi64(z1V, jdx0V, z3V); // 16 17 18 19 20 21 22 23
            __m512i r3V = _mm512_permutex2var_epi64(z1V, jdx1V, z3V); // 24 25 26 27 28 29 30 31

            _mm512_storeu_si512((__m512i*)&b[i + 0], r0V);
            _mm512_storeu_si512((__m512i*)&b[i + 8], r1V);
            _mm512_storeu_si512((__m512i*)&b[i + 16], r2V);
            _mm512_storeu_si512((__m512i*)&b[i + 24], r3V);

        }

        for (size_t i = 0; i < U; i += 2) {
            uint64_t x = b[i];
            uint64_t y = b[i + 1];
            b[i] = x + y;
            b[i + 1] = x - y + p + p;
            b[i] = std::min(b[i], b[i] - 2 * p);
            b[i] = std::min(b[i], b[i] - p);
            b[i + 1] = std::min(b[i + 1], b[i + 1] - 2 * p);
            b[i + 1] = std::min(b[i + 1], b[i + 1] - p);
        }

    }

    template <uint64_t beta_>
    inline void Base3NTT(uint64_t b[]) {
        __m512i x1V = _mm512_set1_epi64(beta_);
        __m512i x2V = _mm512_set1_epi64(Z::Mul(beta_, beta_));
        __m512i pV = _mm512_set1_epi64(p);
        __m512i p2V = _mm512_set1_epi64(p * 2);

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

    }

    inline void BitReverse(uint64_t a[], uint64_t b[]) {
        for (size_t i = 0; i < N; i++) {
            b[i] = a[bit_reverse_table[i]];
        }
    }

    inline void ForwardTensor23NTT(uint64_t a[]) {
        Base3NTT<beta>(a);
        Base2NTT<alpha>(a);
        Base2NTT<alpha>(a + U);
        Base2NTT<alpha>(a + 2 * U);
    }

    inline void InverseTensor23NTT(uint64_t a[]) {
        Base3NTT<beta_inv>(a);
        InverseBase2NTT<alpha_inv>(a);
        InverseBase2NTT<alpha_inv>(a + U);
        InverseBase2NTT<alpha_inv>(a + 2 * U);
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
            omega_table[(N - gi_inv[i]) % N] = t;
            t = Z::Mul(t, omega);
        }

        for (size_t i = 0; i < N; i++) {
            omega_inv_table[i] = omega_table[t0_perm[i]];
        }

        BitReverse(omega_inv_table, omega_table);

        ForwardTensor23NTT(omega_table);
        uint64_t N_inv = Z::Pow(N, p - 2);
        for (size_t i = 0; i < N; i++) {
            omega_inv_table[i] = Z::Inv(omega_table[i]);
            omega_table[i] = Z::Mul(omega_table[i], N_inv);
            omega_inv_table[i] = Z::Mul(omega_inv_table[i], N_inv);
        }
    }

    inline void ForwardNTT(uint64_t a[]) {

        alignas(64) constexpr static std::array<size_t, N> pre_perm = []() {
            std::array<size_t, N> pre_perm;
            for (size_t i = 0; i < N; i++) {
                pre_perm[i] = gi[t0_perm[bit_reverse_table[i]]] - 1;
            }
            return pre_perm;
        }();

        alignas(64) constexpr static std::array<size_t, N> post_perm = []() {
            std::array<size_t, N> post_perm;
            for (size_t i = 0; i < N; i++) {
                post_perm[i] = bit_reverse_table[t0_perm_inv[(N - gi_inv[i + 1]) % N]];
            }
            return post_perm;
        }();

        alignas(64) static uint64_t reg[N];
        
        for (size_t i = 0; i < N; i++) {
            reg[i] = a[pre_perm[i]];
        }

        ForwardTensor23NTT(reg);

        intel::hexl::EltwiseMultMod(reg, reg, omega_table, N, p, 4);

        InverseTensor23NTT(reg);

        for (size_t i = 0; i < N; i++) {
            a[i] = reg[post_perm[i]];
        }
    }

    inline void InverseNTT(uint64_t a[]) {

        alignas(64) constexpr static std::array<size_t, N> pre_perm = []() {
            std::array<size_t, N> pre_perm;
            for (size_t i = 0; i < N; i++) {
                pre_perm[i] = gi[(N - t0_perm[bit_reverse_table[i]]) % N] - 1;
            }
            return pre_perm;
        }();

        alignas(64) constexpr static std::array<size_t, N> post_perm = []() {
            std::array<size_t, N> post_perm;
            for (size_t i = 0; i < N; i++) {
                post_perm[i] = bit_reverse_table[t0_perm_inv[gi_inv[i + 1]]];
            }
            return post_perm;
        }();

        alignas(64) static uint64_t reg[N];

        for (size_t i = 0; i < N; i++) {
            reg[i] = a[pre_perm[i]];
        }

        ForwardTensor23NTT(reg);

        intel::hexl::EltwiseMultMod(reg, reg, omega_inv_table, N, p, 4);

        InverseTensor23NTT(reg);

        for (size_t i = 0; i < N; i++) {
            a[i] = reg[post_perm[i]];
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