#include <cstdint>
#include <cstddef>
#include <array>
#include <iostream>

#include "zp.h"

template <uint64_t p, uint64_t g, size_t O, size_t w>
class NTT {
public:
    using ZZ = Zp<p>;

    constexpr static size_t Compile_Compute_u(size_t N_) {
        size_t u = 0;
        while (N_ % 2 == 0) {
            N_ >>= 1;
            u++;
        }
        return u;
    }

    constexpr static size_t Compile_Compute_U(size_t N_) {
        size_t U = 1 << Compile_Compute_u(N_);
        return U;
    }

    constexpr static size_t Compile_Compute_v(size_t N_) {
        size_t v = 0;
        while (N_ % 3 == 0) {
            N_ /= 3;
            v++;
        }
        return v;
    }

    constexpr static size_t Compile_Compute_V(size_t N_) {
        size_t V = 1;
        for (size_t i = 0; i < Compile_Compute_v(N_); i++) {
            V *= 3;
        }
        return V;
    }

    constexpr static uint64_t ComputeOmegaO() {
        return ZZ::Pow(g, (p - 1) / O);
    }

    constexpr static uint64_t ComputeOmegaOInv() {
        return ZZ::Pow(ComputeOmegaO(), p - 2);
    }

    constexpr static uint64_t ComputeOmegaN() {
        return ZZ::Pow(g, (p - 1) / N);
    }

    constexpr static uint64_t ComputeOmegaNInv() {
        return ZZ::Pow(ComputeOmegaN(), p - 2);
    }

    constexpr static size_t N = O - 1;
    constexpr static size_t u = Compile_Compute_u(N);
    constexpr static size_t U = Compile_Compute_U(N);
    constexpr static size_t v = Compile_Compute_v(N);
    constexpr static size_t V = Compile_Compute_V(N);

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
            omega_N_table[i] = ZZ::Mul(omega_N_table[i - 1], omega_N);
        }
        omega_N_inv_table[0] = 1;
        for (size_t i = 1; i < N; i++) {
            omega_N_inv_table[i] = omega_N_table[N - i];
        }
        for (size_t i = 0; i < N; i++) {
            omega_N_barrett_table[i] = ZZ::ComputeBarrettFactor(omega_N_table[i]);
            omega_N_inv_barrett_table[i] = ZZ::ComputeBarrettFactor(omega_N_inv_table[i]);
        }
    }

    static void CT23NTT(uint64_t a[], uint64_t b[], uint64_t omega[], uint64_t omega_barrett[]) {

        for (size_t i = 0; i < N; i++) {
            b[i] = a[bit_reverse_table[i]];
        }

        for (size_t i = 0; i < N; i += 2) {
            uint64_t t = b[i + 1];
            b[i + 1] = ZZ::Sub(b[i], t);
            b[i] = ZZ::Add(b[i], t);
        }

        size_t l0 = 1, l1 = 2, d = N / 2;

        for (size_t i = 1; i < u; i++) {
            l0 = 1 << i;
            l1 = 1 << (i + 1);
            d = N >> (i + 1);
            for (size_t j = 0; j < N; j += l1) {
                for (size_t k = 0; k < l0; k++) {
                    uint64_t t = ZZ::MulFastConst(b[j + l0 + k], omega[k * d], omega_barrett[k * d]);
                    b[j + l0 + k] = ZZ::Sub(b[j + k], t);
                    b[j + k] = ZZ::Add(b[j + k], t);
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
                for (size_t k = 0; k < l0; k++) {
                    uint64_t y1 = ZZ::MulFastConst(b[j + l0 + k], omega[k * d], omega_barrett[k * d]);
                    uint64_t y2 = ZZ::MulFastConst(b[j + l0 + l0 + k], omega[2 * k * d], omega_barrett[2 * k * d]);
                    uint64_t y0 = ZZ::Add(y1, y2);
                    uint64_t t = ZZ::Add(ZZ::MulFastConst(y1, z3, z3_barrett), ZZ::MulFastConst(y2, zz3, zz3_barrett));
                    b[j + l0 + k] = ZZ::Add(b[j + k], t);
                    b[j + l0 + l0 + k] = ZZ::Sub(b[j + k], ZZ::Add(y0, t));
                    b[j + k] = ZZ::Add(b[j + k], y0);
                }
            }
        }
        std::copy(b, b + N, a);
    }

    static void ForwardCT23NTT(uint64_t a[], uint64_t b[]) {
        CT23NTT(a, b, omega_N_table, omega_N_barrett_table);
    }

    static void InverseCT23NTT(uint64_t a[], uint64_t b[]) {
        CT23NTT(a, b, omega_N_inv_table, omega_N_inv_barrett_table);
    }

    static uint64_t omega_O_table[N];
    static uint64_t omega_O_inv_table[N];
    static uint64_t omega_O_barrett_table[N];
    static uint64_t omega_O_inv_barrett_table[N];

    static void ComputeOmegaOTable() {
        uint64_t t = omega_O;
        for (size_t i = 1; i <= N; i++) {
            omega_O_barrett_table[(N - gi_inv[i]) % N] = t;
            t = ZZ::Mul(t, omega_O);
        }
        ForwardCT23NTT(omega_O_barrett_table, omega_O_table);

        uint64_t N_inv = ZZ::Pow(N, p - 2);
        for (size_t i = 0; i < N; i++) {
            omega_O_inv_table[i] = ZZ::Pow(omega_O_table[i], p - 2);
            omega_O_table[i] = ZZ::Mul(omega_O_table[i], N_inv);
            omega_O_inv_table[i] = ZZ::Mul(omega_O_inv_table[i], N_inv);
            omega_O_barrett_table[i] = ZZ::ComputeBarrettFactor(omega_O_table[i]);
            omega_O_inv_barrett_table[i] = ZZ::ComputeBarrettFactor(omega_O_inv_table[i]);
        }
    }

    static bool initialized;

    NTT() {
        if (!initialized) {
            initialized = true;
            ComputeOmegaNTable();
            ComputeOmegaOTable();
        }
    }

    static void ForwardRaderNTT(uint64_t a[]) {
        static uint64_t reg[N];

        for (size_t i = 0; i < N; i++) {
            reg[i] = a[gi[i] - 1];
        }

        ForwardCT23NTT(reg, a);

        for (size_t i = 0; i < N; i++) {
            a[i] = ZZ::MulFastConst(a[i], omega_O_table[i], omega_O_barrett_table[i]);
        }

        InverseCT23NTT(a, reg);

        for (size_t i = 0; i < N; i++) {
            a[i] = reg[(N - gi_inv[i + 1]) % N];
        }
    }

    static void InverseRaderNTT(uint64_t a[]) {
        static uint64_t reg[N];

        for (size_t i = 0; i < N; i++) {
            reg[i] = a[gi[(N - i) % N] - 1];
        }

        ForwardCT23NTT(reg, a);

        for (size_t i = 0; i < N; i++) {
            a[i] = ZZ::MulFastConst(a[i], omega_O_inv_table[i], omega_O_inv_barrett_table[i]);
        }

        InverseCT23NTT(a, reg);

        for (size_t i = 0; i < N; i++) {
            a[i] = reg[gi_inv[i + 1]];
        }
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

template <uint64_t p, uint64_t g, size_t O, size_t w>
bool NTT<p, g, O, w>::initialized = false;