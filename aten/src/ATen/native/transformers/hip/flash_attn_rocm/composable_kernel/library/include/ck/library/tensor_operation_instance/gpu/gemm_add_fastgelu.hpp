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

void add_device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_mk_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    AddFastGelu>>>&);

void add_device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_mk_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Col,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    AddFastGelu>>>&);

void add_device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Row,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    AddFastGelu>>>&);

void add_device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_km_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Col,
                                                    Row_Tuple,
                                                    Row,
                                                    F16,
                                                    F16,
                                                    F16_Tuple,
                                                    F16,
                                                    PassThrough,
                                                    PassThrough,
                                                    AddFastGelu>>>&);

// GEMM + Add + FastGelu
template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename D0DataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleD<ALayout,
                                                      BLayout,
                                                      ck::Tuple<D0Layout>,
                                                      ELayout,
                                                      ADataType,
                                                      BDataType,
                                                      ck::Tuple<D0DataType>,
                                                      EDataType,
                                                      PassThrough,
                                                      PassThrough,
                                                      AddFastGelu>>
{
    using DeviceOp = DeviceGemmMultipleD<ALayout,
                                         BLayout,
                                         ck::Tuple<D0Layout>,
                                         ELayout,
                                         ADataType,
                                         BDataType,
                                         ck::Tuple<D0DataType>,
                                         EDataType,
                                         PassThrough,
                                         PassThrough,
                                         AddFastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<D0DataType, half_t> && is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<D0Layout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_mk_kn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<D0Layout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_mk_nk_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<D0Layout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_instances(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<D0Layout, Row> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_km_nk_mn_mn_instances(
                    op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
