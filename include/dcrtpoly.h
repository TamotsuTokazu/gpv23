#ifndef DCRTPOLY_H
#define DCRTPOLY_H

#include <cstdint>
#include <concepts>

#include "ntt.h"

template <size_t N>
class Poly {
public:
    int64_t a[N];

    Poly() : a{} {}

    template <std::integral T>
    Poly(std::initializer_list<T> il) {
        size_t i = 0;
        for (auto x : il) {
            a[i++] = x;
            if (i == N) {
                break;
            }
        }
    }

    void print() const {
        for (size_t i = 0; i < N; i++) {
            std::cout << a[i] << " ";
        }
        std::cout << std::endl;
    }
};

template <typename ...NTTs>
class DCRTScalar {
public:
    constexpr static size_t ell = sizeof...(NTTs);
    constexpr static std::array<size_t, ell> p{NTTs::p...};

    uint64_t a[ell];

    DCRTScalar() : a{} {}

    template <std::unsigned_integral T>
    DCRTScalar(T x) {
        for (size_t i = 0; i < ell; i++) {
            a[i] = x % p[i];
        }
    }

    template <std::signed_integral T>
    DCRTScalar(T x) {
        for (size_t i = 0; i < ell; i++) {
            a[i] = (x % (int64_t)p[i] + (int64_t)p[i]) % (int64_t)p[i];
        }
    }
};

template <typename ...NTTs>
class DCRTPoly {
public:
    using NTTTypes = std::tuple<NTTs...>;

    template <size_t i>
    using NTTi = std::tuple_element_t<i, NTTTypes>;

    constexpr static size_t ell = sizeof...(NTTs);
    constexpr static std::array<size_t, ell> p{NTTs::p...};

    static constexpr size_t N = NTTi<0>::N;

    static_assert((... && (NTTs::N == N)), "All NTTs must have the same N");

    uint64_t a[ell][N];

    template <typename F, size_t... Is>
    static void ForEachImpl(F&& f, std::index_sequence<Is...>) {
        (f.template operator()<Is>(), ...);
    }

    template <typename F>
    static void ForEach(F&& f) {
        ForEachImpl(f, std::make_index_sequence<ell>());
    }

    DCRTPoly() : a{} {}

    DCRTPoly(const Poly<N>& poly) {
        for (size_t i = 0; i < N; i++) {
            auto x = DCRTScalar<NTTs...>(poly.a[i]);
            for (size_t j = 0; j < ell; j++) {
                a[j][i] = x.a[j];
            }
        }
        ForEach([this]<size_t i>() {
            using NTT = NTTi<i>;
            NTT::GetInstance().ForwardNTT(a[i]);
        });
    }

    DCRTPoly operator+(const DCRTPoly& rhs) const{
        DCRTPoly res;
        ForEach([this, &rhs, &res]<size_t i>() {
            for (size_t j = 0; j < N; j++) {
                res.a[i][j] = Zp<p[i]>::Add(a[i][j], rhs.a[i][j]);
            }
        });
        return res;
    }

    DCRTPoly operator-(const DCRTPoly& rhs) const {
        DCRTPoly res;
        ForEach([this, &rhs, &res]<size_t i>() {
            for (size_t j = 0; j < N; j++) {
                res.a[i][j] = Zp<p[i]>::Sub(a[i][j], rhs.a[i][j]);
            }
        });
        return res;
    }

    DCRTPoly operator*(const DCRTPoly& rhs) const {
        DCRTPoly res;
        ForEach([this, &rhs, &res]<size_t i>() {
            for (size_t j = 0; j < N; j++) {
                res.a[i][j] = Zp<p[i]>::Mul(a[i][j], rhs.a[i][j]);
            }
        });
        return res;
    }

    DCRTPoly operator*(const DCRTScalar<NTTs...>& rhs) const {
        DCRTPoly res;
        ForEach([this, &rhs, &res]<size_t i>() {
            for (size_t j = 0; j < N; j++) {
                res.a[i][j] = Zp<p[i]>::Mul(a[i][j], rhs.a[i]);
            }
        });
        return res;
    }

    template <typename NTT>
    DCRTPoly Component() const {
        DCRTPoly res;
        ForEach([this, &res]<size_t i>() {
            if constexpr (std::is_same_v<NTT, NTTi<i>>) {
                std::copy(a[i], a[i] + N, res.a[i]);
            }
        });
        return res;
    }

    template <typename NTT>
    Poly<N> Retrieve() const {
        uint64_t t[N];
        ForEach([this, &t]<size_t i>() {
            if constexpr (std::is_same_v<NTT, NTTi<i>>) {
                std::copy(a[i], a[i] + N, t);
                NTT::GetInstance().InverseNTT(t);
            }
        });
        Poly<N> res;
        for (size_t i = 0; i < N; i++) {
            res.a[i] = t[i];
        }
        return res;
    }

    template <typename NTT>
    DCRTPoly BaseExpand() const {
        uint64_t t[N];
        ForEach([this, &t]<size_t i>() {
            if constexpr (std::is_same_v<NTT, NTTi<i>>) {
                std::copy(a[i], a[i] + N, t);
            }
        });
        NTT::GetInstance().InverseNTT(t);
        DCRTPoly res;
        ForEach([this, &res, &t]<size_t i>() {
            using NTTi = NTTi<i>;
            if constexpr (std::is_same_v<NTT, NTTi>) {
                std::copy(a[i], a[i] + N, res.a[i]);
            } else {
                for (size_t j = 0; j < N; j++) {
                    res.a[i][j] = t[j] % p[i];
                }
                NTTi::GetInstance().ForwardNTT(res.a[i]);
            }
        });
        return res;
    }

    void print() const {
        (Retrieve<NTTs>().print(), ...);
    }

    static DCRTPoly SampleUniform() {
        DCRTPoly res;
        ForEach([&res]<size_t i>() {
            for (size_t j = 0; j < N; j++) {
                res.a[i][j] = std::rand() % p[i];
            }
        });
        return res;
    }

    static DCRTPoly SampleE() {
        return SampleUniform();
    }

};

#endif // DCRTPOLY_H