// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "device_conv2d_xdl_int8_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_conv2d_perchannel_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleD<NDimSpatial,
                                                              GNHWC,
                                                              GKYXC,
                                                              GK_Tuple,
                                                              GNHWK,
                                                              int8_t,
                                                              int8_t,
                                                              F32_Tuple,
                                                              int8_t,
                                                              PassThrough,
                                                              PassThrough,
                                                              Mul2_Clamp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_conv2d_int8_32Ds_instances<GK_Tuple, F32_Tuple, Mul2_Clamp, ConvFwdDefault>{});
    add_device_operation_instances(
        instances,
        device_conv2d_int8_32Ds_instances<GK_Tuple, F32_Tuple, Mul2_Clamp, ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_conv2d_int8_32Ds_instances<GK_Tuple, F32_Tuple, Mul2_Clamp, ConvFwd1x1S1P0>{});
}

void add_device_conv2d_relu_perchannel_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleD<NDimSpatial,
                                                              GNHWC,
                                                              GKYXC,
                                                              GK_Tuple,
                                                              GNHWK,
                                                              int8_t,
                                                              int8_t,
                                                              F32_Tuple,
                                                              int8_t,
                                                              PassThrough,
                                                              PassThrough,
                                                              Relu_Mul2_Clamp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_conv2d_int8_32Ds_instances<GK_Tuple, F32_Tuple, Relu_Mul2_Clamp, ConvFwdDefault>{});
    add_device_operation_instances(
        instances,
        device_conv2d_int8_32Ds_instances<GK_Tuple, F32_Tuple, Relu_Mul2_Clamp, ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances,
        device_conv2d_int8_32Ds_instances<GK_Tuple, F32_Tuple, Relu_Mul2_Clamp, ConvFwd1x1S1P0>{});
}
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
