// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Empty_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    Empty_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    FastGelu>>>&);

void add_device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Col,
                                                    Empty_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    Empty_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    FastGelu>>>&);

void add_device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Row,
                                                    Empty_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    Empty_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    FastGelu>>>&);

void add_device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Col,
                                                    Empty_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    Empty_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    FastGelu>>>&);

// GEMM + FastGelu
template <typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGemmMultipleD<ALayout,
                                                                                        BLayout,
                                                                                        Empty_Tuple,
                                                                                        ELayout,
                                                                                        ADataType,
                                                                                        BDataType,
                                                                                        Empty_Tuple,
                                                                                        EDataType,
                                                                                        PassThrough,
                                                                                        PassThrough,
                                                                                        FastGelu>>
{
    using DeviceOp = DeviceGemmMultipleD<ALayout,
                                         BLayout,
                                         Empty_Tuple,
                                         ELayout,
                                         ADataType,
                                         BDataType,
                                         Empty_Tuple,
                                         EDataType,
                                         PassThrough,
                                         PassThrough,
                                         FastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
                add_device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
                add_device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
