#include <iostream>

#include "dcrtpoly.h"
#include "rlwe.h"

using NTT1 = CircNTT<4294960321LL, 13LL, 7, 3>;
using NTT2 = CircNTT<4294953601LL, 37LL, 7, 3>;

int main() {
    Poly<7> a{1, 2, 3, 4, 5, 6};
    DCRTPoly<NTT1, NTT2> aa{a};
    aa.print();
    auto b = aa + aa;
    b.print();
    auto c = aa * aa;
    c.print();
    auto d = aa.BaseExpand<NTT1>();
    d.print();
    auto e = RGSWEncrypt(aa, aa);
    e[0][0][0].print();
    return 0;
}