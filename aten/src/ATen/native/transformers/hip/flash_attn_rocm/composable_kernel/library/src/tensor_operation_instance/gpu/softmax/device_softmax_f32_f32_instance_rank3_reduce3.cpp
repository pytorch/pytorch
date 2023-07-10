// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_rank3_reduce3.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_type.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

static constexpr index_t RANK = 3;

void add_device_softmax_f32_f32_rank3_reduce3_instances(
    std::vector<DeviceSoftmaxPtr<F32, F32, F32, PassThrough, PassThrough, RANK>>& instances)
{
    add_device_operation_instances(instances, device_softmax_f32_f32_instances<RANK, 3>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
