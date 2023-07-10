// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_softmax.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_softmax_i8_i8_rank4_reduce3_instances(
    std::vector<DeviceSoftmaxPtr<I8, F32, I8, PassThrough, PassThrough, 4>>& instances);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
