// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t Rank,
          index_t NumReduceDim,
          typename InElementwiseOperation,
          typename AccElementwiseOperation>
struct DeviceReduce : public BaseOperator
{
    static constexpr index_t NumOutDim = (Rank - NumReduceDim == 0) ? 1 : Rank - NumReduceDim;

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, Rank> inLengths,
                        const std::array<index_t, Rank> inStrides,
                        const std::array<index_t, NumOutDim> outLengths,
                        const std::array<index_t, NumOutDim> outStrides,
                        const std::array<int, NumReduceDim> reduceDims,
                        float alpha,
                        float beta,
                        const void* in_dev,
                        const void* in_index_dev,
                        void* out_dev,
                        void* out_index_dev,
                        const InElementwiseOperation in_elementwise_op,
                        const AccElementwiseOperation acc_elementwise_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <index_t Rank,
          index_t NumReduceDim,
          typename InElementwiseOperation,
          typename AccElementwiseOperation>
using DeviceReducePtr = std::unique_ptr<
    DeviceReduce<Rank, NumReduceDim, InElementwiseOperation, AccElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
