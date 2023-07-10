// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <index_t Rank, index_t Reduce>
using device_softmax_f32_f32_instances = std::tuple<
    // clang-format off
    //                InDataType, AccDataType, OutDataType, InElementwiseOp, AccElementwiseOp, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    DeviceSoftmaxImpl<       F32,         F32,         F32,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  8,                 32,                1,                8,              1,               1,               1>, // fallback kernel
    DeviceSoftmaxImpl<       F32,         F32,         F32,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  8,                 32,                1,                8,              1,               4,               4>,
    DeviceSoftmaxImpl<       F32,         F32,         F32,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  4,                 64,                1,                8,              1,               4,               4>,
    DeviceSoftmaxImpl<       F32,         F32,         F32,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  2,                128,                1,                8,              1,               4,               4>,
    DeviceSoftmaxImpl<       F32,         F32,         F32,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  2,                128,                1,               16,              1,               4,               4>,
    DeviceSoftmaxImpl<       F32,         F32,         F32,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  2,                128,                1,               32,              1,               4,               4>,
    DeviceSoftmaxImpl<       F32,         F32,         F32,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  1,                256,                1,                8,              1,               4,               4>,
    DeviceSoftmaxImpl<       F32,         F32,         F32,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  1,                256,                1,               16,              1,               4,               4>,
    DeviceSoftmaxImpl<       F32,         F32,         F32,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  1,                256,                1,               32,              1,               4,               4>,
    // Reduction on middle dimensions
    // InSrcVectorDim is 0 since we want to coalesce reads on M dimension
    DeviceSoftmaxImpl<       F32,         F32,         F32,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  8,                 32,                8,                4,              0,               1,               1>,
    DeviceSoftmaxImpl<       F32,         F32,         F32,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  8,                 32,                8,                4,              0,               4,               4>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
