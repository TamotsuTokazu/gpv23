#ifndef DCRTPOLY_H
#define DCRTPOLY_H

#include <cstdint>
#include <concepts>

#include "ntt.h"

template <size_t N>
class Poly {
public:
    std::array<int64_t, N> a;

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

    void Print() const {
        for (size_t i = 0; i < N; i++) {
            std::cout << a[i] << " ";
        }
        std::cout << std::endl;
    }
};

template <typename ...Ts>
class TsUtils {
public:
    using Tuple = std::tuple<Ts...>;

    template <size_t i>
    using Type = std::tuple_element_t<i, Tuple>;

    template <typename F, size_t... Is>
    constexpr static void ForEachImpl(F&& f, std::index_sequence<Is...>) {
        (f.template operator()<Is>(), ...);
    }

    template <typename F>
    constexpr static void ForEach(F&& f) {
        ForEachImpl(f, std::make_index_sequence<sizeof...(Ts)>());
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
    constexpr DCRTScalar(T x) {
        TsUtils<NTTs...>::ForEach([this, &x]<size_t i>() {
            a[i] = x % p[i];
        });
    }

    template <std::signed_integral T>
    constexpr DCRTScalar(T x) {
        TsUtils<NTTs...>::ForEach([this, &x]<size_t i>() {
            a[i] = (x % (int64_t)p[i] + (int64_t)p[i]) % p[i];
        });
    }

    constexpr DCRTScalar operator+(const DCRTScalar& rhs) const {
        DCRTScalar res;
        TsUtils<NTTs...>::ForEach([this, &rhs, &res]<size_t i>() {
            res.a[i] = Zp<p[i]>::Add(a[i], rhs.a[i]);
        });
        return res;
    }

    constexpr DCRTScalar operator-(const DCRTScalar& rhs) const {
        DCRTScalar res;
        TsUtils<NTTs...>::ForEach([this, &rhs, &res]<size_t i>() {
            res.a[i] = Zp<p[i]>::Sub(a[i], rhs.a[i]);
        });
        return res;
    }

    constexpr DCRTScalar operator*(const DCRTScalar& rhs) const {
        DCRTScalar res;
        TsUtils<NTTs...>::ForEach([this, &rhs, &res]<size_t i>() {
            res.a[i] = Zp<p[i]>::Mul(a[i], rhs.a[i]);
        });
        return res;
    }

    template <std::unsigned_integral T>
    constexpr DCRTScalar operator*(T x) const {
        DCRTScalar res;
        TsUtils<NTTs...>::ForEach([this, x, &res]<size_t i>() {
            res.a[i] = Zp<p[i]>::Mul(a[i], x % p[i]);
        });
        return res;
    }

    constexpr DCRTScalar Inv() const {
        DCRTScalar res;
        TsUtils<NTTs...>::ForEach([this, &res]<size_t i>() {
            res.a[i] = Zp<p[i]>::Inv(a[i]);
        });
        return res;
    }
};

template <typename ...NTTs>
class DCRTPoly {
public:
    using NTTTypes = std::tuple<NTTs...>;
    using Scalar = DCRTScalar<NTTs...>;

    template <size_t i>
    using NTTi = std::tuple_element_t<i, NTTTypes>;

    template <typename NTT>
    constexpr static size_t index() {
        size_t res = 0;
        U::ForEach([&res]<size_t i>() {
            if constexpr (std::is_same_v<NTT, NTTi<i>>) {
                res = i;
            }
        });
        return res;
    }

    using U = TsUtils<NTTs...>;

    constexpr static size_t ell = sizeof...(NTTs);
    constexpr static std::array<size_t, ell> p{NTTs::p...};

    static constexpr size_t N = NTTi<0>::N;

    static_assert((... && (NTTs::N == N)), "All NTTs must have the same N");

    static constexpr size_t O = NTTi<0>::O;
    static_assert((... && (NTTs::O == O)), "All NTTs must have the same O");

    // D_factors[i][j] = product of p[k] for k != i, k != j mod p[i]
    constexpr static std::array<std::array<uint64_t, ell>, ell> D_factors = [] {
        std::array<std::array<uint64_t, ell>, ell> res{};
        U::ForEach([&res]<size_t i> {
            for (size_t j = 0; j < ell; j++) {
                res[i][j] = 1;
                for (size_t k = 0; k < ell; k++) {
                    if (k != i && k != j) {
                        res[i][j] = Zp<p[i]>::Mul(res[i][j], p[k]);
                    }
                }
            }
        });
        return res;
    }();

    constexpr static std::array<std::array<uint64_t, ell>, ell> D_invs = [] {
        std::array<std::array<uint64_t, ell>, ell> res{};
        U::ForEach([&res]<size_t i> {
            for (size_t j = 0; j < ell; j++) {
                res[i][j] = Zp<p[i]>::Inv(D_factors[i][j]);
            }
        });
        return res;
    }();

    uint64_t a[ell][N];

    DCRTPoly() : a{} {}

    DCRTPoly(const Poly<N>& poly) {
        for (size_t i = 0; i < N; i++) {
            auto x = DCRTScalar<NTTs...>(poly.a[i]);
            for (size_t j = 0; j < ell; j++) {
                a[j][i] = x.a[j];
            }
        }
        U::ForEach([this]<size_t i>() {
            NTTi<i>::GetInstance().ForwardNTT(a[i]);
        });
    }

    DCRTPoly operator+(const DCRTPoly& rhs) const{
        DCRTPoly res;
        U::ForEach([this, &rhs, &res]<size_t i> {
            for (size_t j = 0; j < N; j++) {
                res.a[i][j] = Zp<p[i]>::Add(a[i][j], rhs.a[i][j]);
            }
        });
        return res;
    }

    DCRTPoly operator-(const DCRTPoly& rhs) const {
        DCRTPoly res;
        U::ForEach([this, &rhs, &res]<size_t i> {
            for (size_t j = 0; j < N; j++) {
                res.a[i][j] = Zp<p[i]>::Sub(a[i][j], rhs.a[i][j]);
            }
        });
        return res;
    }

    DCRTPoly operator*(const DCRTPoly& rhs) const {
        DCRTPoly res;
        U::ForEach([this, &rhs, &res]<size_t i> {
            for (size_t j = 0; j < N; j++) {
                res.a[i][j] = Zp<p[i]>::Mul(a[i][j], rhs.a[i][j]);
            }
        });
        return res;
    }

    DCRTPoly operator*(const DCRTScalar<NTTs...>& rhs) const {
        DCRTPoly res;
        U::ForEach([this, &rhs, &res]<size_t i> {
            for (size_t j = 0; j < N; j++) {
                res.a[i][j] = Zp<p[i]>::Mul(a[i][j], rhs.a[i]);
            }
        });
        return res;
    }

    template <typename NTT>
    DCRTPoly Component() const {
        DCRTPoly res;
        constexpr size_t id = index<NTT>();
        std::copy(a[id], a[id] + N, res.a[id]);
        return res;
    }

    template <typename NTT>
    Poly<N> Retrieve() const {
        uint64_t t[N];
        constexpr size_t id = index<NTT>();
        std::copy(a[id], a[id] + N, t);
        NTT::GetInstance().InverseNTT(t);
        Poly<N> res;
        for (size_t i = 0; i < N; i++) {
            res.a[i] = t[i];
        }
        return res;
    }

    template <typename NTT>
    DCRTPoly BaseExtend() const {
        uint64_t t[N];
        constexpr size_t id = index<NTT>();
        std::copy(a[id], a[id] + N, t);
        NTT::GetInstance().InverseNTT(t);
        DCRTPoly res;
        U::ForEach([this, &res, &t]<size_t i> {
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

    template <typename NTT>
    Poly<N> ModulusSwitch() const {
        uint64_t result[N];
        constexpr size_t id = index<NTT>();
        std::copy(a[id], a[id] + N, result);
        uint64_t factor = 1;
        NTT::GetInstance().InverseNTT(result);
        U::ForEach([this, &result, id, &factor]<size_t i>() {
            if constexpr (!std::is_same_v<NTT, NTTi<i>>) {
                factor = NTT::Z::Mul(factor, p[i]);
                uint64_t temp[N];
                std::copy(a[i], a[i] + N, temp);
                NTTi<i>::GetInstance().InverseNTT(temp);
                for (size_t j = 0; j < N; j++) {
                    temp[j] = Zp<p[i]>::Mul(temp[j], D_invs[i][id]);
                    temp[j] = Zp<p[id]>::Mul(temp[j], D_factors[id][i]);
                    result[j] = Zp<p[id]>::Sub(result[j], temp[j] % p[id]);
                }
            }
        });
        factor = NTT::Z::Inv(factor);
        for (size_t i = 0; i < N; i++) {
            result[i] = NTT::Z::Mul(result[i], factor);
        }
        Poly<N> res;
        std::copy(result, result + N, res.a.data());
        return res;
    }

    void Print() const {
        (Retrieve<NTTs>().Print(), ...);
    }

    static DCRTPoly SampleUniform() {
        DCRTPoly res;
        U::ForEach([&res]<size_t i> {
            for (size_t j = 0; j < N; j++) {
                res.a[i][j] = std::rand() % p[i];
            }
        });
        return res;
    }

    static DCRTPoly SampleE() {
        return DCRTPoly();
    }

    constexpr DCRTPoly Galois(size_t alpha) const requires (N == O) {
        DCRTPoly res;
        U::ForEach([this, &res, alpha]<size_t i>() {
            for (size_t j = 0; j < N; j++) {
                res.a[i][j] = a[i][((uint64_t)j * alpha) % N];
            }
        });
        return res;
    }

    constexpr DCRTPoly Galois(size_t alpha) const requires (N == O - 1) {
        DCRTPoly res;
        U::ForEach([this, &res, alpha]<size_t i>() {
            for (size_t j = 1; j <= N; j++) {
                res.a[i][j - 1] = a[i][((uint64_t)j * alpha) % N - 1];
            }
        });
        return res;
    }

    constexpr static DCRTPoly Monomial(size_t e, std::signed_integral auto a) requires (N == O) {
        Poly<N> poly;
        poly.a[e % O] = a;
        return DCRTPoly(poly);
    }

};

#endif // DCRTPOLY_H