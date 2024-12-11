#include "ntt.h"

using T = NTT23<562896521097217LL, 5LL, 97, 5>;

int main() {
    std::cout << T::omega << " " << T::Z::Mul(T::omega, T::omega) << std::endl;
    std::array<uint64_t, T::N> a;
    for (size_t i = 0; i < T::N; i++) {
        a[i] = i;
    }
    T::GetInstance().ForwardNTT(a.data());
    for (size_t i = 0; i < T::N; i++) {
        a[i] %= T::p;
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
    T::GetInstance().InverseNTT(a.data());
    for (size_t i = 0; i < T::N; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}