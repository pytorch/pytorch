// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "device_conv2d_xdl_int8_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_conv2d_perlayer_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleD<NDimSpatial,
                                                              GNHWC,
                                                              GKYXC,
                                                              Empty_Tuple,
                                                              GNHWK,
                                                              int8_t,
                                                              int8_t,
                                                              Empty_Tuple,
                                                              int8_t,
                                                              PassThrough,
                                                              PassThrough,
                                                              Mul_Clamp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_conv2d_int8_instances<Empty_Tuple, Empty_Tuple, Mul_Clamp, ConvFwdDefault>{});
    add_device_operation_instances(
        instances,
        device_conv2d_int8_instances<Empty_Tuple, Empty_Tuple, Mul_Clamp, ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_conv2d_int8_instances<Empty_Tuple, Empty_Tuple, Mul_Clamp, ConvFwd1x1S1P0>{});
}

void add_device_conv2d_relu_perlayer_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleD<NDimSpatial,
                                                              GNHWC,
                                                              GKYXC,
                                                              Empty_Tuple,
                                                              GNHWK,
                                                              int8_t,
                                                              int8_t,
                                                              Empty_Tuple,
                                                              int8_t,
                                                              PassThrough,
                                                              PassThrough,
                                                              Relu_Mul_Clamp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_conv2d_int8_instances<Empty_Tuple, Empty_Tuple, Relu_Mul_Clamp, ConvFwdDefault>{});
    add_device_operation_instances(
        instances,
        device_conv2d_int8_instances<Empty_Tuple, Empty_Tuple, Relu_Mul_Clamp, ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_conv2d_int8_instances<Empty_Tuple, Empty_Tuple, Relu_Mul_Clamp, ConvFwd1x1S1P0>{});
}
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
