// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/utility/reduction_enums.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <ck::ReduceTensorOp ReduceOpId>
struct DevicePool2dFwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* in_dev,
                        void* out_dev,
                        void* out_indices_dev,
                        ck::index_t N,
                        ck::index_t C,
                        std::array<ck::index_t, 2> input_spatial_lengths,
                        std::array<ck::index_t, 2> window_spatial_lengths,
                        std::array<ck::index_t, 2> output_spatial_lengths,
                        std::array<ck::index_t, 2> window_strides,
                        std::array<ck::index_t, 2> input_left_pads,
                        std::array<ck::index_t, 2> input_right_pads) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <ck::ReduceTensorOp ReduceOpId>
using DevicePool2dFwdPtr = std::unique_ptr<DevicePool2dFwd<ReduceOpId>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
