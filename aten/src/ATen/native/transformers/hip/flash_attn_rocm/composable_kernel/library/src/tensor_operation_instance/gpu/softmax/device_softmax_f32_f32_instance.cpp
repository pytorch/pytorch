// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>

#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_rank3_reduce1.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_rank3_reduce2.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_rank3_reduce3.hpp"

#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_rank4_reduce1.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_rank4_reduce2.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_rank4_reduce3.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_rank4_reduce4.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_softmax_f32_f32_rank3_instances(
    std::vector<DeviceSoftmaxPtr<F32, F32, F32, PassThrough, PassThrough, 3>>& instances)
{
    add_device_softmax_f32_f32_rank3_reduce1_instances(instances);
    add_device_softmax_f32_f32_rank3_reduce2_instances(instances);
    add_device_softmax_f32_f32_rank3_reduce3_instances(instances);
}

void add_device_softmax_f32_f32_rank4_instances(
    std::vector<DeviceSoftmaxPtr<F32, F32, F32, PassThrough, PassThrough, 4>>& instances)
{
    add_device_softmax_f32_f32_rank4_reduce1_instances(instances);
    add_device_softmax_f32_f32_rank4_reduce2_instances(instances);
    add_device_softmax_f32_f32_rank4_reduce3_instances(instances);
    add_device_softmax_f32_f32_rank4_reduce4_instances(instances);
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
