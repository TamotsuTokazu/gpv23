#include <iostream>

#include "dcrtpoly.h"

using NTT1 = NTT<4294960321LL, 13LL, 7, 3>;
using NTT2 = NTT<4294953601LL, 37LL, 7, 3>;

int main() {
    Poly<6> a{1, 2, 3, 4, 5, 6};
    DCRTPoly<NTT1, NTT2> aa{a};
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 6; j++) {
            std::cout << aa.a[i][j] << " ";
        }
        std::cout << std::endl;
    }
    auto b = aa + aa;
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 6; j++) {
            std::cout << b.a[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}