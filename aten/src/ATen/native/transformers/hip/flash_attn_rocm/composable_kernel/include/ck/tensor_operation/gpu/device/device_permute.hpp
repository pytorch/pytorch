// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>
#include <memory>
#include <type_traits>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NumDim, typename InDataType, typename OutDataType, typename ElementwiseOperation>
struct DevicePermute : BaseOperator
{
    using Lengths = std::array<index_t, NumDim>;
    using Strides = Lengths;

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const Lengths& in_lengths,
                        const Strides& in_strides,
                        const Lengths& out_lengths,
                        const Strides& out_strides,
                        const void* in_dev_buffer,
                        void* out_dev_buffer,
                        ElementwiseOperation elementwise_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
