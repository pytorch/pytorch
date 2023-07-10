// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "test_softmax_util.hpp"

class TestSoftmaxInterface : public ::testing::Test
{
    protected:
    template <ck::index_t Rank, ck::index_t NumReduceDims>
    using SoftmaxInstance =
        ck::DeviceSoftmaxInstanceWrapper<Rank, NumReduceDims, 256, 1, 256, 1, 8, 1, 8, 8>;
};

TEST_F(TestSoftmaxInterface, IncorrectReduceDims)
{
    std::vector<ck::index_t> lengths{2, 128, 1536};
    std::vector<ck::index_t> strides{128 * 1536, 1536, 1};

    EXPECT_THROW((SoftmaxInstance<3, 1>{}.IsSupported(lengths, strides, {-1})), std::runtime_error);
    EXPECT_THROW((SoftmaxInstance<3, 1>{}.IsSupported(lengths, strides, {3})), std::runtime_error);
    EXPECT_THROW((SoftmaxInstance<3, 1>{}.IsSupported(lengths, strides, {0, 1})),
                 std::runtime_error);
    EXPECT_THROW((SoftmaxInstance<3, 1>{}.IsSupported(lengths, strides, {})), std::runtime_error);
    EXPECT_THROW((SoftmaxInstance<3, 2>{}.IsSupported(lengths, strides, {2, -1})),
                 std::runtime_error);
    EXPECT_THROW((SoftmaxInstance<3, 2>{}.IsSupported(lengths, strides, {2, 4})),
                 std::runtime_error);
    EXPECT_THROW((SoftmaxInstance<3, 2>{}.IsSupported(lengths, strides, {2})), std::runtime_error);
}

TEST_F(TestSoftmaxInterface, IncorrectLengthsSize)
{
    std::vector<ck::index_t> lengths{128, 1536};
    std::vector<ck::index_t> strides{128 * 1536, 1536, 1};
    std::vector<ck::index_t> reduce_dims{2};

    EXPECT_THROW((SoftmaxInstance<3, 1>{}.IsSupported({128, 1536}, strides, reduce_dims)),
                 std::runtime_error);
    EXPECT_THROW((SoftmaxInstance<3, 1>{}.IsSupported({}, strides, reduce_dims)),
                 std::runtime_error);
    EXPECT_THROW((SoftmaxInstance<3, 1>{}.IsSupported({1, 8, 128, 1536}, strides, reduce_dims)),
                 std::runtime_error);
}

TEST_F(TestSoftmaxInterface, IncorrectStridesSize)
{
    std::vector<ck::index_t> lengths{2, 128, 1536};
    std::vector<ck::index_t> reduce_dims{2};

    EXPECT_THROW((SoftmaxInstance<3, 1>{}.IsSupported(lengths, {1536, 1}, reduce_dims)),
                 std::runtime_error);
    EXPECT_THROW((SoftmaxInstance<3, 1>{}.IsSupported(lengths, {}, reduce_dims)),
                 std::runtime_error);
    EXPECT_THROW((SoftmaxInstance<3, 1>{}.IsSupported(lengths, {1, 2, 3, 4}, reduce_dims)),
                 std::runtime_error);
}

TEST_F(TestSoftmaxInterface, UnsupportedLengths)
{
    using SoftmaxInstance1 = ck::DeviceSoftmaxInstanceWrapper<3, 1, 256, 1, 256, 1, 8, 1, 8, 4>;
    EXPECT_FALSE(SoftmaxInstance1{}.IsSupported({2, 128, 1500}, {128 * 1500, 1500, 1}, {2}));
    EXPECT_FALSE(SoftmaxInstance1{}.IsSupported({2, 127, 1536}, {127 * 1536, 1536, 1}, {2}));
    EXPECT_FALSE(SoftmaxInstance1{}.IsSupported({2, 128, 1537}, {128 * 1537, 1537, 1}, {2}));

    // Reduction of middle dimensions
    using SoftmaxInstance2 = ck::DeviceSoftmaxInstanceWrapper<3, 3, 256, 8, 32, 8, 8, 0, 8, 4>;
    EXPECT_FALSE(SoftmaxInstance2{}.IsSupported({2, 128, 1536}, {128 * 1536, 1536, 1}, {0, 1, 2}));

    // Reduction of middle dimensions
    using SoftmaxInstance3 = ck::DeviceSoftmaxInstanceWrapper<3, 1, 256, 8, 32, 8, 8, 0, 4, 8>;
    EXPECT_FALSE(SoftmaxInstance3{}.IsSupported({2, 128, 1536}, {128 * 1536, 1536, 1}, {2}));
    EXPECT_FALSE(SoftmaxInstance3{}.IsSupported({2, 128, 1537}, {128 * 1537, 1537, 1}, {1}));
    EXPECT_FALSE(SoftmaxInstance3{}.IsSupported({2, 128, 1540}, {128 * 1540, 1540, 1}, {1}));
    EXPECT_FALSE(SoftmaxInstance3{}.IsSupported({2, 127, 1536}, {127 * 1536, 1536, 1}, {1}));
}

TEST_F(TestSoftmaxInterface, UnsupportedInstance)
{
    // Instance with InSrcVectorDim = 1, can't reduce middle dims if in/out vec size != 1
    using SoftmaxInstance1 = ck::DeviceSoftmaxInstanceWrapper<3, 1, 256, 8, 32, 1, 8, 1, 8, 8>;
    EXPECT_FALSE(SoftmaxInstance1{}.IsSupported({2, 128, 1024}, {128 * 1024, 1024, 1}, {0}));
}
