#ifndef ZP_H
#define ZP_H

#include <cstdint>
#include <immintrin.h>
#include <limits>

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

    constexpr static inline uint64_t Inv(uint64_t x) {
        return Pow(x, p - 2);
    }

    constexpr static double u = (1.0 + std::numeric_limits<double>::epsilon()) / (double)p;

    static inline __m512i Mul512(__m512i x, __m512i y) requires (p < (1ULL << 50)) {
        static __m512d v_u = _mm512_set1_pd(u);
        static __m512d q = _mm512_set1_pd(p);
        const auto rounding = _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC;
        __m512d x_d = _mm512_cvt_roundepu64_pd(x, rounding);
        __m512d y_d = _mm512_cvt_roundepu64_pd(y, rounding);
        __m512d h = _mm512_mul_pd(x_d, y_d);
        __m512d l = _mm512_fmsub_pd(x_d, y_d, h);
        __m512d b = _mm512_mul_pd(h, v_u);
        __m512d c = _mm512_floor_pd(b);
        __m512d d = _mm512_fnmadd_pd(c, q, h);
        __m512d g = _mm512_add_pd(d, l);
        __mmask8 m = _mm512_cmp_pd_mask(g, _mm512_setzero_pd(), _CMP_LT_OQ);
        g = _mm512_mask_add_pd(g, m, g, q);
        __m512i z = _mm512_cvt_roundpd_epi64(g, rounding);
        return z;
    }

};

#endif // ZP_H