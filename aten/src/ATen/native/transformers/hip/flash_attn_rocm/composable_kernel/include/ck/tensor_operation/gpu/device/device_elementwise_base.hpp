// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <memory>
#include <array>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          index_t NumDim>
struct DeviceElementwiseBase : public BaseOperator
{
    static constexpr int NumInput  = InDataTypeTuple::Size();
    static constexpr int NumOutput = OutDataTypeTuple::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, NumDim> lengths,
                        const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                        const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                        const std::array<const void*, NumInput> in_dev_buffers,
                        const std::array<void*, NumOutput> out_dev_buffers,
                        ElementwiseOperation elementwise_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
}; // namespace device

template <typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          index_t NumDim>
using DeviceElementwiseBasePtr = std::unique_ptr<
    DeviceElementwiseBase<InDataTypeTuple, OutDataTypeTuple, ElementwiseOperation, NumDim>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
