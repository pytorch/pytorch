// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_batched_gemm_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmGemm<Row,
                                                      Col,
                                                      Row,
                                                      Row,
                                                      F16,
                                                      F16,
                                                      F16,
                                                      F16,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough>>>& instances);

void add_device_batched_gemm_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gon_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmGemm<Row,
                                                      Col,
                                                      Col,
                                                      Row,
                                                      F16,
                                                      F16,
                                                      F16,
                                                      F16,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough>>>& instances);
template <typename ALayout,
          typename B0Layout,
          typename B1Layout,
          typename CLayout,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceBatchedGemmGemm<ALayout,
                                                        B0Layout,
                                                        B1Layout,
                                                        CLayout,
                                                        ADataType,
                                                        B0DataType,
                                                        B1DataType,
                                                        CDataType,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough>>
{
    using DeviceOp = DeviceBatchedGemmGemm<ALayout,
                                           B0Layout,
                                           B1Layout,
                                           CLayout,
                                           ADataType,
                                           B0DataType,
                                           B1DataType,
                                           CDataType,
                                           PassThrough,
                                           PassThrough,
                                           PassThrough,
                                           PassThrough,
                                           PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, half_t> && is_same_v<B0DataType, half_t> &&
                     is_same_v<B1DataType, half_t> && is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<B0Layout, Col> &&
                         is_same_v<B1Layout, Row> && is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
                    op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<B0Layout, Col> &&
                              is_same_v<B1Layout, Col> && is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gon_gmo_instance(
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
