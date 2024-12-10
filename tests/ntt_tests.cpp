#include <gtest/gtest.h>

#include "ntt.h"

TEST(NTTInitTests, InitSmall) {
    using NTT = NTT<1152921504606846577LL, 15, 7, 3>;
    EXPECT_EQ(NTT::N, 6);
    EXPECT_EQ(NTT::u, 1);
    EXPECT_EQ(NTT::U, 2);
    EXPECT_EQ(NTT::v, 1);
    EXPECT_EQ(NTT::V, 3);
}

TEST(NTTInitTests, InitLarge) {
    using NTT = NTT<1152921504107839489LL, 19, 12289, 11>;
    EXPECT_EQ(NTT::N, 12288);
    EXPECT_EQ(NTT::u, 12);
    EXPECT_EQ(NTT::U, 4096);
    EXPECT_EQ(NTT::v, 1);
    EXPECT_EQ(NTT::V, 3);
}

TEST(NTTInitTests, GiSmall) {
    using NTT = NTT<1152921504606846577LL, 15, 7, 3>;
    EXPECT_EQ(NTT::gi, (std::array<uint64_t, 6>{1, 3, 2, 6, 4, 5}));
}

TEST(NTTInitTests, GiInvSmall) {
    using NTT = NTT<1152921504606846577LL, 15, 7, 3>;
    EXPECT_EQ(NTT::gi_inv, (std::array<uint64_t, 7>{(uint64_t)-1, 0, 2, 1, 4, 5, 3}));
}

TEST(NTT23Tests, Small) {
    using NTT = NTT23<562896521097217LL, 5LL, 97, 5>;
    EXPECT_EQ(NTT::N, 96);
    EXPECT_EQ(NTT::u, 5);
    EXPECT_EQ(NTT::U, 32);

    std::array<uint64_t, NTT::N> a = {1};
    NTT::GetInstance().ForwardTensor23NTT(a.data());

    for (size_t i = 0; i < NTT::N; i++) {
        EXPECT_EQ(a[i], 1);
    }
}
