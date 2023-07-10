// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include <array>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/utility/reduction_enums.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t Rank,
          index_t NumReduceDim,
          index_t NumReduction,
          typename InElementwiseOperationTuple,
          typename AccElementwiseOperationTuple>
struct DeviceMultipleReduce : public BaseOperator
{
    static constexpr index_t NumInputDim  = Rank;
    static constexpr index_t NumOutputDim = (Rank - NumReduceDim > 1) ? Rank - NumReduceDim : 1;

    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const std::array<index_t, NumInputDim> inLengths,
        const std::array<index_t, NumInputDim> inStrides,
        const std::array<index_t, NumOutputDim> outLengths,
        const std::array<std::array<index_t, NumOutputDim>, NumReduction> outStrides,
        const std::array<int, NumReduceDim> reduceDims,
        const std::array<const void*, NumReduction> alphas,
        const std::array<const void*, NumReduction> betas,
        const void* in_dev,
        const std::array<void*, NumReduction> out_dev_buffers,
        const InElementwiseOperationTuple in_elementwise_op_tuple,
        const AccElementwiseOperationTuple acc_elementwise_op_tuple) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <index_t Rank,
          index_t NumReduceDim,
          index_t NumReduction,
          typename InElementwiseOperationTuple,
          typename AccElementwiseOperationTuple>
using DeviceMultipleReducePtr = std::unique_ptr<DeviceMultipleReduce<Rank,
                                                                     NumReduceDim,
                                                                     NumReduction,
                                                                     InElementwiseOperationTuple,
                                                                     AccElementwiseOperationTuple>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
