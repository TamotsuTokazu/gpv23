#include <iostream>

#include "ntt.h"

int main() {
    using NTT = NTT<1152921504606846577LL, 15, 7, 3>;
    NTT();
    uint64_t a[NTT::N] = {1, 2, 3, 0, 0, 0};
    uint64_t b[NTT::N] = {1, 0, 0, 1, 0, 0};
    NTT::ForwardRaderNTT(a);
    NTT::ForwardRaderNTT(b);
    for (size_t i = 0; i < NTT::N; i++) {
        std::cout << a[i] << " ";
        a[i] = NTT::ZZ::Mul(a[i], b[i]);
    }
    std::cout << std::endl;
    NTT::InverseRaderNTT(a);
    for (size_t i = 0; i < NTT::N; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}