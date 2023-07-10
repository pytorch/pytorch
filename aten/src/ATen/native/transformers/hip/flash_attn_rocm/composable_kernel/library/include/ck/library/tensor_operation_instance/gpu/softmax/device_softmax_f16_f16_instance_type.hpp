// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <index_t Rank, index_t Reduce>
using device_softmax_f16_f16_instances = std::tuple<
    // clang-format off
    //                InDataType, AccDataType, OutDataType, InElementwiseOp, AccElementwiseOp, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    // fallback kernel
    DeviceSoftmaxImpl<       F16,         F32,         F16,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  8,                 32,                1,                8,              1,               1,              1>,
    DeviceSoftmaxImpl<       F16,         F32,         F16,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  8,                 32,                1,                8,              1,               8,              8>,
    DeviceSoftmaxImpl<       F16,         F32,         F16,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  4,                 64,                1,                8,              1,               8,              8>,
    DeviceSoftmaxImpl<       F16,         F32,         F16,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  2,                128,                1,                8,              1,               8,              8>,
    DeviceSoftmaxImpl<       F16,         F32,         F16,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  2,                128,                1,               16,              1,               8,              8>,
    DeviceSoftmaxImpl<       F16,         F32,         F16,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  2,                128,                1,               32,              1,               8,              8>,
    DeviceSoftmaxImpl<       F16,         F32,         F16,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  1,                256,                1,                8,              1,               8,              8>,
    DeviceSoftmaxImpl<       F16,         F32,         F16,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  1,                256,                1,               16,              1,               8,              8>,
    DeviceSoftmaxImpl<       F16,         F32,         F16,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  1,                256,                1,               32,              1,               8,              8>,
    // Reduction on middle dimensions
    // InSrcVectorDim is 0 since we want to coalesce reads on M dimension
    DeviceSoftmaxImpl<       F16,         F32,         F16,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  8,                 32,                8,                4,              0,               1,              1>,
    DeviceSoftmaxImpl<       F16,         F32,         F16,     PassThrough,      PassThrough, Rank,       Reduce,       256,                  8,                 32,                8,                4,              0,               8,              4>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
