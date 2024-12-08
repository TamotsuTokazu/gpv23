#ifndef ZP_H
#define ZP_H

#include <cstdint>
#include <immintrin.h>
#include <limits>

template <uint64_t p_>
class Zp {
public:

    constexpr static uint64_t p = p_;
    constexpr static size_t bitshift = 64;

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
        uint64_t q = (uint64_t)((__uint128_t)a * b_mu >> bitshift);
        uint64_t prod = a * b - q * p;
        return prod >= p ? prod - p : prod;
    }

    constexpr static uint64_t ComputeBarrettFactor(uint64_t x) {
        return (uint64_t)(((__uint128_t(x) << bitshift) / p));
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

    static inline __m512i _mm512_hexl_mulhi_epi(__m512i x, __m512i y) {
        // https://stackoverflow.com/questions/28807341/simd-signed-with-unsigned-multiplication-for-64-bit-64-bit-to-128-bit
        __m512i lo_mask = _mm512_set1_epi64(0x00000000ffffffff);
        // Shuffle high bits with low bits in each 64-bit integer =>
        // x0_lo, x0_hi, x1_lo, x1_hi, x2_lo, x2_hi, ...
        __m512i x_hi = _mm512_shuffle_epi32(x, (_MM_PERM_ENUM)0xB1);
        // y0_lo, y0_hi, y1_lo, y1_hi, y2_lo, y2_hi, ...
        __m512i y_hi = _mm512_shuffle_epi32(y, (_MM_PERM_ENUM)0xB1);
        __m512i z_lo_lo = _mm512_mul_epu32(x, y);        // x_lo * y_lo
        __m512i z_lo_hi = _mm512_mul_epu32(x, y_hi);     // x_lo * y_hi
        __m512i z_hi_lo = _mm512_mul_epu32(x_hi, y);     // x_hi * y_lo
        __m512i z_hi_hi = _mm512_mul_epu32(x_hi, y_hi);  // x_hi * y_hi

        //                   x_hi | x_lo
        // x                 y_hi | y_lo
        // ------------------------------
        //                  [x_lo * y_lo]    // z_lo_lo
        // +           [z_lo * y_hi]         // z_lo_hi
        // +           [x_hi * y_lo]         // z_hi_lo
        // +    [x_hi * y_hi]                // z_hi_hi
        //     ^-----------^ <-- only bits needed
        //  sum_|  hi | mid | lo  |

        // Low bits of z_lo_lo are not needed
        __m512i z_lo_lo_shift = _mm512_srli_epi64(z_lo_lo, 32);

        //                   [x_lo  *  y_lo] // z_lo_lo
        //          + [z_lo  *  y_hi]        // z_lo_hi
        //          ------------------------
        //            |    sum_tmp   |
        //            |sum_mid|sum_lo|
        __m512i sum_tmp = _mm512_add_epi64(z_lo_hi, z_lo_lo_shift);
        __m512i sum_lo = _mm512_and_si512(sum_tmp, lo_mask);
        __m512i sum_mid = _mm512_srli_epi64(sum_tmp, 32);
        //            |       |sum_lo|
        //          + [x_hi   *  y_lo]       // z_hi_lo
        //          ------------------
        //            [   sum_mid2   ]
        __m512i sum_mid2 = _mm512_add_epi64(z_hi_lo, sum_lo);
        __m512i sum_mid2_hi = _mm512_srli_epi64(sum_mid2, 32);
        __m512i sum_hi = _mm512_add_epi64(z_hi_hi, sum_mid);
        return _mm512_add_epi64(sum_hi, sum_mid2_hi);
    }

    static inline __m512i MulConst512(__m512i x, __m512i y, __m512i y_mu) {
        __m512i xyV = _mm512_mullo_epi64(x, y);
        __m512i qV = _mm512_hexl_mulhi_epi(x, y_mu);
        const __m512i pV = _mm512_set1_epi64(p);
        qV = _mm512_mullo_epi64(qV, pV);
        __m512i subV = _mm512_sub_epi64(xyV, qV);
        return _mm512_min_epu64(subV, _mm512_sub_epi64(subV, pV));
    }

};

#endif // ZP_H