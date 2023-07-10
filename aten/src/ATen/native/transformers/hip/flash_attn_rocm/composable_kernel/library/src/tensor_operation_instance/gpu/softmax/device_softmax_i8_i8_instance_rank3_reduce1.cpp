// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_i8_i8_instance_rank3_reduce1.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_i8_i8_instance_type.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

static constexpr index_t RANK = 3;

void add_device_softmax_i8_i8_rank3_reduce1_instances(
    std::vector<DeviceSoftmaxPtr<I8, F32, I8, PassThrough, PassThrough, RANK>>& instances)
{
    add_device_operation_instances(instances, device_softmax_i8_i8_instances<RANK, 1>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
