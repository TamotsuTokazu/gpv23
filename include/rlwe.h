#ifndef RLWE_H
#define RLWE_H

#include "dcrtpoly.h"

template <typename DCRT>
using RLWECiphertext = std::array<DCRT, 2>;

template <typename DCRT>
RLWECiphertext<DCRT> RLWEEncrypt(const DCRT& m, const DCRT& s) {
    auto a = DCRT::SampleUniform();
    auto e = DCRT::SampleE();
    return {a, a * s + m + e};
}

template <typename DCRT>
using RLWEGadgetCiphertext = std::array<RLWECiphertext<DCRT>, DCRT::ell>;

template <typename DCRT>
using RGSWCiphertext = std::array<RLWEGadgetCiphertext<DCRT>, 2>;

template <typename DCRT>
RLWEGadgetCiphertext<DCRT> RLWEGadgetEncrypt(const DCRT& m, const DCRT& s) {
    RLWEGadgetCiphertext<DCRT> c;
    DCRT::ForEach([&c, &m, &s]<size_t i>() {
        c[i] = RLWEEncrypt(m.template Component<typename DCRT::NTTi<i>>(), s);
    });
    return c;
}

template <typename DCRT>
RGSWCiphertext<DCRT> RGSWEncrypt(const DCRT& m, const DCRT& s) {
    return {RLWEGadgetEncrypt(m, s), RLWEGadgetEncrypt(m * s, s)};
}

template <typename DCRT>
RLWECiphertext<DCRT> ExtMult(const RLWECiphertext<DCRT>& c, const RGSWCiphertext<DCRT>& C) {
    RLWECiphertext<DCRT> res;
    DCRT::ForEach([&res, &c, &C]<size_t i>() {
        auto c0i = c[0].template Component<typename DCRT::NTTi<i>>();
        auto c1i = c[1].template Component<typename DCRT::NTTi<i>>();
        res[0] = res[0] + C[0][i][0] * c0i - C[1][i][0] * c1i;
        res[1] = res[1] + C[0][i][1] * c0i - C[1][i][1] * c1i;
    });
    return res;
}

#endif // RLWE_H