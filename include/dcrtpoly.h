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
    constexpr static size_t ell = sizeof...(NTTs);
    constexpr static std::array<size_t, ell> p{NTTs::p...};

    static constexpr size_t N = std::tuple_element<0, std::tuple<NTTs...>>::type::N;

    static_assert((... && (NTTs::N == N)), "All NTTs must have the same N");

    uint64_t a[ell][N];

    template <typename NTT, size_t i>
    struct ToEval {
        void operator()(DCRTPoly *t) {
            NTT::GetInstance().ForwardNTT(t->a[i]);
        }
    };

    template <typename NTT, size_t i>
    struct Plus {
        void operator()(const DCRTPoly *t, const DCRTPoly& x, DCRTPoly &ret) {
            for (size_t j = 0; j < N; j++) {
                ret.a[i][j] = Zp<p[i]>::Add(t->a[i][j], x.a[i][j]);
            }
        }
    };

    template <typename NTT, size_t i>
    struct Minus {
        void operator()(const DCRTPoly *t, const DCRTPoly& x, DCRTPoly &ret) {
            for (size_t j = 0; j < N; j++) {
                ret.a[i][j] = Zp<p[i]>::Sub(t->a[i][j], x.a[i][j]);
            }
        }
    };

    template <typename NTT, size_t i>
    struct Times {
        void operator()(const DCRTPoly *t, const DCRTPoly& x, DCRTPoly &ret) {
            for (size_t j = 0; j < N; j++) {
                ret.a[i][j] = Zp<p[i]>::Mul(t->a[i][j], x.a[i][j]);
            }
        }
    };

    template <template <typename, size_t> typename F, size_t ...Is>
    void UnaryExecFImpl(DCRTPoly *t, std::index_sequence<Is...>) {
        (F<NTTs, Is>()(t), ...);
    }

    template <template <typename, size_t> typename F>
    void UnaryExecF() {
        UnaryExecFImpl<F>(this, std::make_index_sequence<ell>());
    }

    template <typename T, template <typename, size_t> typename F, size_t ...Is>
    void TrinaryExecFConstImpl(const DCRTPoly *t, const T &x, T &ret, std::index_sequence<Is...>) const {
        (F<NTTs, Is>()(t, x, ret), ...);
    }

    template <typename T, template <typename, size_t> typename F>
    void TrinaryExecFConst(const T &x, T &ret) const {
        TrinaryExecFConstImpl<T, F>(this, x, ret, std::make_index_sequence<ell>());
    }

    DCRTPoly() : a{} {}

    DCRTPoly(const Poly<N>& poly) {
        for (size_t i = 0; i < N; i++) {
            auto x = DCRTScalar<NTTs...>(poly.a[i]);
            for (size_t j = 0; j < ell; j++) {
                a[j][i] = x.a[j];
            }
        }
        UnaryExecF<ToEval>();
    }

    DCRTPoly operator+(const DCRTPoly& rhs) const {
        DCRTPoly res;
        TrinaryExecFConst<DCRTPoly, Plus>(rhs, res);
        return res;
    }

    DCRTPoly operator-(const DCRTPoly& rhs) const {
        DCRTPoly res;
        TrinaryExecFConst<DCRTPoly, Minus>(rhs, res);
        return res;
    }

    DCRTPoly operator*(const DCRTPoly& rhs) const {
        DCRTPoly res;
        TrinaryExecFConst<DCRTPoly, Times>(rhs, res);
        return res;
    }

};


#endif // DCRTPOLY_H