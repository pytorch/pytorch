// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <int BlockSize, int MThreadClusterSize, int KThreadClusterSize>
struct ReductionConfiguration_1
{
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize, "Invalid Configuration!");

    static constexpr int BlockSize_          = BlockSize;
    static constexpr int MThreadClusterSize_ = MThreadClusterSize;
    static constexpr int KThreadClusterSize_ = KThreadClusterSize;
};

template <int InSrcVectorDim,
          int InSrcVectorSize,
          int OutDstVectorSize,
          int MThreadSliceSize,
          int KThreadSliceSize>
struct ReductionConfiguration_2
{
    static constexpr int InSrcVectorDim_   = InSrcVectorDim;
    static constexpr int InSrcVectorSize_  = InSrcVectorSize;
    static constexpr int OutDstVectorSize_ = OutDstVectorSize;
    static constexpr int MThreadSliceSize_ = MThreadSliceSize;
    static constexpr int KThreadSliceSize_ = KThreadSliceSize;
};

using ReduceAdd  = ck::reduce::Add;
using ReduceMin  = ck::reduce::Min;
using ReduceMax  = ck::reduce::Max;
using ReduceAMax = ck::reduce::AMax;

using UnarySquare = ck::tensor_operation::element_wise::UnarySquare;
using UnarySqrt   = ck::tensor_operation::element_wise::UnarySqrt;
using UnaryDivide = ck::tensor_operation::element_wise::UnaryDivide;
using UnaryAbs    = ck::tensor_operation::element_wise::UnaryAbs;

#define QUICK_REDUCE_TEST 1

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
