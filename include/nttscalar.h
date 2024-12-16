#ifndef NTTAVX256_H
#define NTTAVX256_H

#include <cstdint>
#include <cstddef>
#include <array>
#include <iostream>

#include <immintrin.h>
#include <hexl/hexl.hpp>

#include "zp.h"

template <uint64_t p_, uint64_t g_, size_t O_, size_t w_>
class NTT23Scalar {

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
    inline void Radix2NTT(uint64_t b[]) {
        constexpr static std::array<uint64_t, U> alpha_table = []() {
            std::array<uint64_t, U> alpha_table;
            alpha_table[0] = 1;
            for (size_t i = 1; i < U; i++) {
                alpha_table[i] = Z::Mul(alpha_table[i - 1], alpha_);
            }
            return alpha_table;
        }();

        constexpr static std::array<uint64_t, U> alpha_barrett_table = []() {
            std::array<uint64_t, U> alpha_barrett_table;
            for (size_t i = 0; i < U; i++) {
                alpha_barrett_table[i] = Z::ComputeBarrettFactor(alpha_table[i]);
            }
            return alpha_barrett_table;
        }();

        for (size_t i = 0; i < U; i += 2) {
            uint64_t t = b[i + 1];
            b[i + 1] = Z::Sub(b[i], t);
            b[i] = Z::Add(b[i], t);
        }

        for (size_t i = 1; i < u; i++) {
            size_t l0 = 1 << i;
            size_t l1 = 1 << (i + 1);
            size_t d = U >> (i + 1);

            for (size_t j = 0; j < U; j += l1) {
                for (size_t k = 0; k < l0; k++) {

                    uint64_t omega = alpha_table[k * d];
                    uint64_t omega_barrett = alpha_barrett_table[k * d];

                    uint64_t t = Z::MulFastConst(b[j + k + l0], omega, omega_barrett);
                    b[j + k + l0] = Z::Sub(b[j + k], t);
                    b[j + k] = Z::Add(b[j + k], t);
                }
            }
        }
    }

    template <uint64_t alpha_>
    inline void InverseRadix2NTT(uint64_t b[]) {
        constexpr static std::array<uint64_t, U> alpha_table = []() {
            std::array<uint64_t, U> alpha_table;
            alpha_table[0] = 1;
            for (size_t i = 1; i < U; i++) {
                alpha_table[i] = Z::Mul(alpha_table[i - 1], alpha_);
            }
            return alpha_table;
        }();

        constexpr static std::array<uint64_t, U> alpha_barrett_table = []() {
            std::array<uint64_t, U> alpha_barrett_table;
            for (size_t i = 0; i < U; i++) {
                alpha_barrett_table[i] = Z::ComputeBarrettFactor(alpha_table[i]);
            }
            return alpha_barrett_table;
        }();

        for (size_t i = u - 1; i >= 1; i--) {
            size_t l0 = 1 << i;
            size_t l1 = 1 << (i + 1);
            size_t d = U >> (i + 1);

            for (size_t j = 0; j < U; j += l1) {
                for (size_t k = 0; k < l0; k++) {
                    
                    uint64_t omega = alpha_table[k * d];
                    uint64_t omega_barrett = alpha_barrett_table[k * d];

                    uint64_t t = Z::Sub(b[j + k], b[j + k + l0]);
                    b[j + k] = Z::Add(b[j + k], b[j + k + l0]);
                    b[j + k + l0] = Z::MulFastConst(t, omega, omega_barrett);
                }
            }
        }

        for (size_t i = 0; i < U; i += 2) {
            uint64_t t = b[i + 1];
            b[i + 1] = Z::Sub(b[i], t);
            b[i] = Z::Add(b[i], t);
        }
    }

    template <uint64_t beta_>
    inline void Radix3NTT(uint64_t b[]) {
        constexpr uint64_t x1 = beta_;
        constexpr uint64_t x2 = Z::Mul(beta_, beta_);
        constexpr uint64_t x1_barrett = Z::ComputeBarrettFactor(x1);
        constexpr uint64_t x2_barrett = Z::ComputeBarrettFactor(x2);

        for (size_t i = 0; i < U; i++) {
            uint64_t y1 = Z::MulFastConst(b[i + U], x1, x1_barrett);
            uint64_t y2 = Z::MulFastConst(b[i + 2 * U], x2, x2_barrett);

            uint64_t t0 = Z::Add(b[i + U], b[i + 2 * U]);
            uint64_t t1 = Z::Add(y1, y2);
            uint64_t t2 = Z::Sub(0, Z::Add(t0, t1));

            b[i + U] = Z::Add(b[i], t1);
            b[i + 2 * U] = Z::Add(b[i], t2);
            b[i] = Z::Add(b[i], t0);
        }

    }

    inline void BitReverse(uint64_t a[], uint64_t b[]) {
        for (size_t i = 0; i < N; i++) {
            b[i] = a[bit_reverse_table[i]];
        }
    }

    inline void ForwardTensor23NTT(uint64_t a[]) {
        Radix3NTT<beta>(a);
        Radix2NTT<alpha>(a);
        Radix2NTT<alpha>(a + U);
        Radix2NTT<alpha>(a + 2 * U);
    }

    inline void InverseTensor23NTT(uint64_t a[]) {
        Radix3NTT<beta_inv>(a);
        InverseRadix2NTT<alpha_inv>(a);
        InverseRadix2NTT<alpha_inv>(a + U);
        InverseRadix2NTT<alpha_inv>(a + 2 * U);
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
    alignas(64) uint64_t omega_barrett_table[N];
    alignas(64) uint64_t omega_inv_barrett_table[N];

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
            omega_barrett_table[i] = Z::ComputeBarrettFactor(omega_table[i]);
            omega_inv_barrett_table[i] = Z::ComputeBarrettFactor(omega_inv_table[i]);
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

        for (size_t i = 0; i < N; i++) {
            reg[i] = Z::MulFastConst(reg[i], omega_table[i], omega_barrett_table[i]);
        }

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

        for (size_t i = 0; i < N; i++) {
            reg[i] = Z::MulFastConst(reg[i], omega_inv_table[i], omega_inv_barrett_table[i]);
        }

        InverseTensor23NTT(reg);

        for (size_t i = 0; i < N; i++) {
            a[i] = reg[post_perm[i]];
        }
    }

    static inline NTT23Scalar &GetInstance() {
        static NTT23Scalar instance;
        return instance;
    }

private:

    NTT23Scalar() {
        ComputeOmegaOTable();
    }

};

#endif // NTTAVX256_H