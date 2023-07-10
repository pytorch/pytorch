// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "test_batched_gemm_gemm_util.hpp"

template <typename Tuple>
class TestBatchedGemmGemmFP16 : public TestBatchedGemmGemm<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<F16, F16, F16, F16, Row, Col, Row, Row>,
    std::tuple<F16, F16, F16, F16, Row, Col, Col, Row>
    >;
// clang-format on

TYPED_TEST_SUITE(TestBatchedGemmGemmFP16, KernelTypes);

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16) { this->Run(); }

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadM)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {136, 128, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadN)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 136, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadK)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 40, 128, 1},
        {128, 128, 136, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_PadO)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 136, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddM)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {129, 128, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddN)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 129, 32, 128, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddK)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 33, 128, 1},
        {128, 128, 129, 128, 1},
    };
    this->Run();
}

// If kernel B1Layout is RowMajor, expect not to support odd O size
TYPED_TEST(TestBatchedGemmGemmFP16, Test_FP16_OddO)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {128, 128, 32, 129, 1},
    };
    this->Run();
}

TYPED_TEST(TestBatchedGemmGemmFP16, DISABLED_Bench_FP16)
{
    this->lengths_ = std::vector<std::vector<int>>{
        {256, 256, 64, 64, 768},
        {256, 256, 128, 128, 768},
        {512, 512, 64, 64, 768},
        {512, 512, 128, 128, 768},
        {1024, 1024, 64, 64, 768},
        {1024, 1024, 128, 128, 768},
        {2048, 2048, 64, 64, 768},
        {2048, 2048, 128, 128, 768},
        {4096, 4096, 64, 64, 768},
        {4096, 4096, 128, 128, 768},
    };
    this->bench_  = true;
    this->verify_ = false;
    this->Run();
}

using ck::tensor_operation::device::GemmSpecialization;

TEST(TestBatchedGemmGemmInterface, GemmSpecializationSizeMatch)
{
    int P = 120; // requires padding
    int Q = 128; // do not require padding

    // IsSupported(M, N, K, O)
    // clang-format off
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::Default>{}.IsSupported(Q, Q, Q, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MPadding>{}.IsSupported(P, Q, Q, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::NPadding>{}.IsSupported(Q, P, Q, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::KPadding>{}.IsSupported(Q, Q, P, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNPadding>{}.IsSupported(P, P, Q, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MKPadding>{}.IsSupported(P, Q, P, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::NKPadding>{}.IsSupported(Q, P, P, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKPadding>{}.IsSupported(P, P, P, Q));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::OPadding>{}.IsSupported(Q, Q, Q, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MOPadding>{}.IsSupported(P, Q, Q, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::NOPadding>{}.IsSupported(Q, P, Q, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::KOPadding>{}.IsSupported(Q, Q, P, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNOPadding>{}.IsSupported(P, P, Q, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MKOPadding>{}.IsSupported(P, Q, P, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::NKOPadding>{}.IsSupported(Q, P, P, P));
    EXPECT_TRUE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKOPadding>{}.IsSupported(P, P, P, P));
    // clang-format on
}

TEST(TestBatchedGemmGemmInterface, GemmSpecializationSizeMismatch)
{
    // IsSupported(M, N, K, O)
    // clang-format off
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::Default>{}.IsSupported(128, 128, 120, 128));
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKPadding>{}.IsSupported(128, 128, 128, 120));
    // Kernel can't support odd K size because SrcVectorDim == KDim and must satisfy SizeKRaw % ABSrcScalarPerVector == 0
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKOPadding>{}.IsSupported(128, 128, 129, 128));
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKOPadding>{}.IsSupported(128, 128, 130, 128));
    // Kernel can't support odd O size because SrcVectorDim == ODim and must satisfy SizeORaw % B1SrcScalarPerVector == 0
    EXPECT_FALSE(DeviceInstanceWrapper_TNTT_FP16_M128_N128_K32_O128<GemmSpecialization::MNKOPadding>{}.IsSupported(128, 128, 128, 129));
    // clang-format on
}
