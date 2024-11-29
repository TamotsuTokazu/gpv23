#ifndef ZP_H
#define ZP_H

#include <cstdint>

template <uint64_t p_>
class Zp {
public:

    constexpr static uint64_t p = p_;

    constexpr static uint64_t Add(uint64_t a, uint64_t b) {
        uint64_t sum = a + b;
        return sum >= p ? sum - p : sum;
    }

    constexpr static uint64_t Sub(uint64_t a, uint64_t b) {
        return a >= b ? a - b : p - b + a;
    }

    constexpr static uint64_t Mul(uint64_t a, uint64_t b) {
        return (uint64_t)((__uint128_t)a * b % p);
    }

    constexpr static uint64_t MulFastConst(uint64_t a, uint64_t b, uint64_t b_mu) {
        uint64_t q = (uint64_t)((__uint128_t)a * b_mu >> 64);
        uint64_t prod = a * b - q * p;
        return prod >= p ? prod - p : prod;
    }

    constexpr static uint64_t ComputeBarrettFactor(uint64_t x) {
        return (uint64_t)(((__uint128_t(x) << 64) / p));
    }

    constexpr static uint64_t Pow(uint64_t x, uint64_t e) {
        uint64_t res = 1;
        uint64_t base = x;

        while (e > 0) {
            if (e & 1) {
                res = Mul(res, base);
            }
            base = Mul(base, base);
            e >>= 1;
        }
        return res;
    }
};

#endif // ZP_H