#include <iostream>
#include <vector>

#include "dcrtpoly.h"
#include "rlwe.h"

#include <chrono>

#define START_TIMER start = std::chrono::system_clock::now()
#define END_TIMER std::cout << "Time: " << (std::chrono::duration<double>(std::chrono::system_clock::now() - start).count()) << std::endl

std::chrono::time_point<std::chrono::system_clock> start;

constexpr size_t p = 12289;
constexpr size_t gp = 11;
// constexpr size_t p = 17;
// constexpr size_t gp = 3;

constexpr size_t N = 512;
constexpr size_t Ncyc = N * 2;

using NTT1 = CircNTT<562936689020929LL, 7LL, p, gp>;
using NTT2 = CircNTT<562918719160321LL, 14LL, p, gp>;
using NTT3 = CircNTT<562880212316161LL, 14LL, p, gp>;
using NTT4 = CircNTT<562851973963777LL, 5LL, p, gp>;

using DCRT = DCRTPoly<NTT1, NTT2, NTT3, NTT4>;

using VecN = std::array<uint64_t, N>;
using Z = Zp<p>;

constexpr size_t rN = Z::Pow(gp, (p - 1) / Ncyc);
constexpr size_t rho = 16;
constexpr size_t Rx = 32;
constexpr uint64_t zeta = Z::Pow(rN, rho);

VecN PartialFourierTransform(VecN a, size_t rho) {
    size_t n = a.size();
    for (size_t i = n; i > rho; i >>= 1) {
        size_t j = i >> 1;
        size_t nn = n / i;
        uint64_t z = Z::Pow(rN, j); // par::rN.ModExp(j, par::p);
        uint64_t w = Z::Pow(rN, i); // par::rN.ModExp(i, par::p);
        for (size_t k = 0, kk = 0; k < nn; k++) {
            for (size_t l = 0; l < j; l++) {
                uint64_t u = Z::Mul(a[kk * i + l + j], z);
                a[kk * i + l + j] = Z::Sub(a[kk * i + l], u);
                // a[kk * i + l].ModAddEq(u, par::p);
                a[kk * i + l] = Z::Add(a[kk * i + l], u);
            }
            for (size_t l = nn >> 1; l > (kk ^= l); l >>= 1);
            // z.ModMulEq(w, par::p);
            z = Z::Mul(z, w);
        }
    }
    return a;
}

VecN PartialInverseFourierTransform(VecN a, size_t rho, size_t r) {
    size_t n = a.size();
    std::vector<uint64_t> temp(r);
    std::vector<size_t> index;

    for (size_t i = rho; i < n; i *= r) {
        size_t j = std::min(i * r, n);
        r = j / i;
        uint64_t W = Z::Pow(rN, Ncyc - Ncyc / (j / rho));
        uint64_t w = Z::Pow(rN, Ncyc - Ncyc / r);
        if (r != index.size()) {
            index.resize(r);
            for (size_t k = 0, kk = 0; k < r; k++) {
                index[k] = kk;
                for (size_t l = r >> 1; l > (kk ^= l); l >>= 1);
            }
        }
        for (size_t k = 0; k < n; k += j) {
            uint64_t z = 1;
            for (size_t l = 0; l < i; l += rho) {
                for (size_t m = 0; m < rho; m++) {
                    uint64_t zz = z;
                    for (size_t f = 0; f < r; f++) {
                        auto &t = temp[f];
                        uint64_t zzz = zz;
                        t = a[k + l + m];
                        for (size_t g = 1; g < r; g++) {
                            t = Z::Add(t, Z::Mul(a[k + l + m + index[g] * i], zzz));
                            zzz = Z::Mul(zzz, zz);
                        }
                        // zz.ModMulEq(w, par::p);
                        zz = Z::Mul(zz, w);
                    }
                    for (size_t f = 0; f < r; f++) {
                        a[k + l + m + f * i] = temp[f];
                    }
                }
                // z.ModMulEq(W, par::p);
                z = Z::Mul(z, W);
            }
        }
    }
    // Integer w = par::rN.ModExp(par::Ncyc - rho, par::p);
    // Integer z = 1;
    uint64_t w = Z::Pow(rN, Ncyc - rho);
    uint64_t z = 1;
    for (size_t i = 0; i < n; i += rho) {
        for (size_t j = 0; j < rho; j++) {
            // a[i + j].ModMulEq(z, par::p);
            a[i + j] = Z::Mul(a[i + j], z);
        }
        // z.ModMulEq(w, par::p);
        z = Z::Mul(z, w);
    }
    // return std::move(a);
    return a;
}

using GaloisKey = std::array<RLWEGadgetCiphertext<DCRT>, p>;

GaloisKey GaloisKeyGen(const DCRT &s) {
    GaloisKey Kg;
    for (size_t i = 2; i < p; i++) {
        Kg[i] = KeySwitchGen(s.Galois(i), s);
    }
    return Kg;
}

RLWEGadgetCiphertext<DCRT> EvalInnerProduct(const GaloisKey &Kg, RLWEGadgetCiphertext<DCRT> ct, std::vector<RGSWCiphertext<DCRT>>::iterator z, const std::vector<uint64_t> &a, size_t l, size_t stride) {
    uint64_t t = 1;
    for (size_t i = 0; i < l; i++) {
        if (a[i] != 0) {
            t = Z::Mul(t, Z::Inv(a[i]));
            if (t != 1) {
                ct = Galois(ct, t);
                ct = KeySwitch(ct, Kg[t]);
            }
            ct = ExtMult(ct, z[i * stride]);
            t = a[i];
        }
    }
    if (t != 1) {
        ct = Galois(ct, t);
        ct = KeySwitch(ct, Kg[t]);
    }
    return ct;
}

std::vector<RLWEGadgetCiphertext<DCRT>> HomomorphicPFT(const GaloisKey &Kg, const RLWEGadgetCiphertext<DCRT> Kss, std::vector<RLWEGadgetCiphertext<DCRT>> z, DCRT s) {
    std::vector<RGSWCiphertext<DCRT>> regs(N);
    std::vector<size_t> index;

    for (size_t i = rho; i < N; i *= Rx) {
        size_t j = std::min(i * Rx, N);
        size_t r = j / i;
        uint64_t W = Z::Pow(rN, Ncyc - Ncyc / (j / rho));
        uint64_t w = Z::Pow(rN, Ncyc - Ncyc / r);
        if (r != index.size()) {
            index.resize(r);
            for (size_t k = 0, kk = 0; k < r; k++) {
                index[k] = kk;
                for (size_t l = r >> 1; l > (kk ^= l); l >>= 1);
            }
        }
        for (size_t k = 0; k < N; k++) {
            regs[k] = SchemeSwitch(z[k], Kss);
        }
        for (size_t k = 0; k < N; k += j) {
            uint64_t Z = 1;
            for (size_t l = 0; l < i; l += rho) {
                for (size_t m = 0; m < rho; m++) {
                    uint64_t zz = Z;
                    for (size_t f = 0; f < r; f++) {
                        uint64_t zzz = zz;
                        std::vector<uint64_t> a(r - 1);
                        for (size_t g = 1; g < r; g++) {
                            a[index[g] - 1] = zzz;
                            zzz = Z::Mul(zzz, zz);
                        }
                        z[k + l + m + f * i] = EvalInnerProduct(Kg, regs[k + l + m][1], regs.begin() + k + l + m + i, a, r - 1, i);
                        zz = Z::Mul(zz, w);
                    }
                }
                Z = Z::Mul(Z, W);
            }
        }
    }

    uint64_t w = Z::Pow(rN, Ncyc - rho);
    uint64_t Z = 1;
    for (size_t i = 0; i < N; i += rho) {
        for (size_t j = 0; j < rho; j++) {
            if (Z != 1) {
                z[i + j] = Galois(z[i + j], Z);
                z[i + j] = KeySwitch(z[i + j], Kg[Z]);
            }
        }
        Z = Z::Mul(Z, w);
    }

    return z;
}

int main() {
    VecN a, b0, z;
    for (size_t i = 0; i < N; i++) {
        z[i] = rand() % p;
        a[i] = rand() % p;
        b0[i] = 0;
    }
    for (size_t i = 0; i < N; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < N; i++) {
        std::cout << z[i] << " ";
    }
    std::cout << std::endl;
    VecN b = b0;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (i >= j) {
                b[i] = Z::Add(b[i], Z::Mul(a[j], z[i - j]));
            } else {
                b[i] = Z::Sub(b[i], Z::Mul(a[j], z[i + N - j]));
            }
        }
    }

    for (size_t i = 0; i < N; i++) {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;

    Poly<p> s_poly;
    for (size_t i = 0; i < N; i++) {
        s_poly.a[i] = rand() % 3 - 1;
    }
    DCRT s(s_poly);

    GaloisKey Kg = GaloisKeyGen(s);
    RLWEGadgetCiphertext<DCRT> Kss = RLWEGadgetEncrypt(s * s, s);

    VecN aP = PartialFourierTransform(a, rho);
    VecN zP = PartialFourierTransform(z, rho);

    for (size_t i = 0; i < N; i++) {
        aP[i] = Z::Mul(aP[i], Z::Inv(N / rho));
    }

    std::vector<RGSWCiphertext<DCRT>> zz(N);
    for (size_t i = 0; i < N; i++) {
        zz[i] = RGSWEncrypt(DCRT::Monomial(zP[i], 1), s);
    }

    START_TIMER;

    RLWEGadgetCiphertext empty_reg = RLWEGadgetEncrypt(DCRT::Monomial(0, 1), s);

    std::vector<RLWEGadgetCiphertext<DCRT>> regs(N);
    uint64_t zzeta = zeta;
    for (size_t k = 0, kk = 0; k < N / rho; k++) {
        for (size_t i = 0; i < rho; i++) {
            std::vector<uint64_t> a(rho);
            for (size_t j = 0; j <= i; j++) {
                a[i - j] = aP[kk * rho + j];
            }
            for (size_t j = i + 1; j < rho; j++) {
                a[rho - j + i] = Z::Mul(aP[kk * rho + j], zzeta);
            }
            regs[kk * rho + i] = EvalInnerProduct(Kg, empty_reg, zz.begin() + kk * rho, a, rho, 1);
        }
        zzeta = Z::Mul(zzeta, zeta);
        zzeta = Z::Mul(zzeta, zeta);
        for (size_t l = N / (2 * rho); l > (kk ^= l); l >>= 1);
    }

    END_TIMER;

    START_TIMER;

    std::vector<RLWEGadgetCiphertext<DCRT>> zPFT = HomomorphicPFT(Kg, Kss, regs, s);

    END_TIMER;

    for (size_t i = 0; i < N; i++) {
        std::cout << DecryptAndPrintE(zPFT[i], s) << " ";
    }
    std::cout << std::endl;

    return 0;
}