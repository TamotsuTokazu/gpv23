#ifndef POLY_H
#define POLY_H

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <iostream>

#include "ntt.h"

template <typename NTT_>
class Poly {
public:
    using NTT = NTT_;
    static const NTT ntt;

    static constexpr uint64_t p = NTT::p;
    static constexpr size_t O = NTT::O;
    static constexpr size_t N = NTT::N;

    using Z = Zp<p>;

    bool is_coeff = false;
    uint64_t a[N];

    Poly(bool is_coeff_ = true) : is_coeff(is_coeff_), a{} {}

    void ToCoeff() {
        if (is_coeff) {
            return;
        }
        NTT::GetInstance().InverseNTT(a);
        is_coeff = true;
    }

    void ToNTT() {
        if (!is_coeff) {
            return;
        }
        NTT::GetInstance().ForwardNTT(a);
        is_coeff = false;
    }

    Poly operator+(const Poly &rhs) const {
        if (is_coeff != rhs.is_coeff) {
            throw std::runtime_error("Addition is not supported between coefficient and NTT domain");
        }

        Poly ret(is_coeff);
        for (size_t i = 0; i < N; i++) {
            ret.a[i] = Z::Add(a[i], rhs.a[i]);
        }
        return ret;
    }

    Poly operator-(const Poly &rhs) const {
        if (is_coeff != rhs.is_coeff) {
            throw std::runtime_error("Addition is not supported between coefficient and NTT domain");
        }

        Poly ret(is_coeff);
        for (size_t i = 0; i < N; i++) {
            ret.a[i] = Z::Sub(a[i], rhs.a[i]);
        }
        return ret;
    }

    Poly operator*(const Poly &rhs) const {
        if (is_coeff || rhs.is_coeff) {
            throw std::runtime_error("Multiplication is not supported in coefficient domain");
        }

        Poly ret(is_coeff);
        for (size_t i = 0; i < N; i++) {
            ret.a[i] = Z::Mul(a[i], rhs.a[i]);
        }
        return ret;
    }

    Poly operator*(uint64_t rhs) const {
        Poly ret(is_coeff);
        for (size_t i = 0; i < N; i++) {
            ret.a[i] = Z::Mul(a[i], rhs);
        }
        return ret;
    }

    void print() const {
        if (is_coeff) {
            std::cout << "Coeff: ";
        } else {
            std::cout << "NTT: ";
        }
        for (size_t i = 0; i < N; i++) {
            std::cout << a[i] << " ";
        }
        std::cout << std::endl;
    }

    template <typename T>
    requires std::signed_integral<T>
    static Poly FromCoeff(const std::vector<T> &v) {
        Poly ret(true);
        for (size_t i = 0; i < N && i < v.size(); i++) {
            if (v[i] < 0) {
                ret.a[i] = v[i] % (int64_t)p + p;
            } else {
                ret.a[i] = v[i] % p;
            }
        }
        return ret;
    }

    template <typename T>
    requires std::unsigned_integral<T>
    static Poly FromCoeff(const std::vector<T> &v) {
        Poly ret(true);
        for (size_t i = 0; i < N && i < v.size(); i++) {
            ret.a[i] = v[i] % p;
        }
        return ret;
    }

    constexpr static Poly GaloisConjugate(const Poly &x, const size_t &a) {
    Poly ret(x.is_coeff);
    if (x.is_coeff) {
        ret.a[a] = x.a[0];
        for (size_t i = 1; i < Poly::N; i++) {
            ret.a[i * a % Poly::O] = x.a[i];
        }
    } else {
        ret.a[0] = x.a[0];
        for (size_t i = 1; i < Poly::N; i++) {
            ret.a[i] = x.a[i * a % Poly::O];
        }
    }
    return ret;
}
};

#endif // POLY_H