// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "test_softmax_util.hpp"

template <ck::index_t N>
using I = ck::Number<N>;

using F16 = ck::half_t;
using F32 = float;
using I8  = int8_t;

template <typename Tuple>
class TestSoftmax : public ck::TestSoftmax<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    //         InDataType, AccDataType, OutDataType, Rank
    std::tuple<       F16,         F32,         F16,    I<3>>,
    std::tuple<       F32,         F32,         F32,    I<3>>,
    std::tuple<        I8,         F32,          I8,    I<3>>
    >;
// clang-format on

TYPED_TEST_SUITE(TestSoftmax, KernelTypes);

#include "test_softmax_ut_cases.inc"
