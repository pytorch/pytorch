// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_softmax.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_softmax_f16_f16_rank3_instances(
    std::vector<DeviceSoftmaxPtr<F16, F32, F16, PassThrough, PassThrough, 3>>&);
void add_device_softmax_f16_f16_rank4_instances(
    std::vector<DeviceSoftmaxPtr<F16, F32, F16, PassThrough, PassThrough, 4>>&);

void add_device_softmax_f32_f32_rank3_instances(
    std::vector<DeviceSoftmaxPtr<F32, F32, F32, PassThrough, PassThrough, 3>>&);
void add_device_softmax_f32_f32_rank4_instances(
    std::vector<DeviceSoftmaxPtr<F32, F32, F32, PassThrough, PassThrough, 4>>&);

void add_device_softmax_i8_i8_rank3_instances(
    std::vector<DeviceSoftmaxPtr<I8, F32, I8, PassThrough, PassThrough, 3>>&);
void add_device_softmax_i8_i8_rank4_instances(
    std::vector<DeviceSoftmaxPtr<I8, F32, I8, PassThrough, PassThrough, 4>>&);

template <typename InDataType, typename AccDataType, typename OutDataType, index_t Rank>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::
        DeviceSoftmax<InDataType, AccDataType, OutDataType, PassThrough, PassThrough, Rank>>
{
    using DeviceOp =
        DeviceSoftmax<InDataType, AccDataType, OutDataType, PassThrough, PassThrough, Rank>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(std::is_same_v<InDataType, F16> && std::is_same_v<AccDataType, F32> &&
                     std::is_same_v<OutDataType, F16>)
        {
            if constexpr(Rank == 3)
                add_device_softmax_f16_f16_rank3_instances(op_ptrs);
            else if constexpr(Rank == 4)
                add_device_softmax_f16_f16_rank4_instances(op_ptrs);
        }
        else if constexpr(std::is_same_v<InDataType, F32> && std::is_same_v<AccDataType, F32> &&
                          std::is_same_v<OutDataType, F32>)
        {
            if constexpr(Rank == 3)
                add_device_softmax_f32_f32_rank3_instances(op_ptrs);
            else if constexpr(Rank == 4)
                add_device_softmax_f32_f32_rank4_instances(op_ptrs);
        }
        else if constexpr(std::is_same_v<InDataType, I8> && std::is_same_v<AccDataType, F32> &&
                          std::is_same_v<OutDataType, I8>)
        {
            if constexpr(Rank == 3)
                add_device_softmax_i8_i8_rank3_instances(op_ptrs);
            else if constexpr(Rank == 4)
                add_device_softmax_i8_i8_rank4_instances(op_ptrs);
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
