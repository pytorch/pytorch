// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_multiple_d_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

using CDE0ElementOp = ck::tensor_operation::element_wise::AddRelu;
using CDE1ElementOp = ck::tensor_operation::element_wise::Add;

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_batched_gemm_add_relu_gemm_add_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultipleDGemmMultipleD<Row,
                                                                        Col,
                                                                        ck::Tuple<Row>,
                                                                        Row,
                                                                        ck::Tuple<Row>,
                                                                        Row,
                                                                        F16,
                                                                        F16,
                                                                        ck::Tuple<F16>,
                                                                        F16,
                                                                        ck::Tuple<F16>,
                                                                        F16,
                                                                        PassThrough,
                                                                        PassThrough,
                                                                        CDE0ElementOp,
                                                                        PassThrough,
                                                                        CDE1ElementOp>>>&
        instances);

void add_device_batched_gemm_add_relu_gemm_add_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gon_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmMultipleDGemmMultipleD<Row,
                                                                        Col,
                                                                        ck::Tuple<Row>,
                                                                        Col,
                                                                        ck::Tuple<Row>,
                                                                        Row,
                                                                        F16,
                                                                        F16,
                                                                        ck::Tuple<F16>,
                                                                        F16,
                                                                        ck::Tuple<F16>,
                                                                        F16,
                                                                        PassThrough,
                                                                        PassThrough,
                                                                        CDE0ElementOp,
                                                                        PassThrough,
                                                                        CDE1ElementOp>>>&
        instances);

template <typename A0Layout,
          typename B0Layout,
          typename D0sLayout,
          typename B1Layout,
          typename D1sLayout,
          typename E1Layout,
          typename A0DataType,
          typename B0DataType,
          typename D0sDataType,
          typename B1DataType,
          typename D1sDataType,
          typename E1DataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceBatchedGemmMultipleDGemmMultipleD<A0Layout,
                                                                          B0Layout,
                                                                          D0sLayout,
                                                                          B1Layout,
                                                                          D1sLayout,
                                                                          E1Layout,
                                                                          A0DataType,
                                                                          B0DataType,
                                                                          D0sDataType,
                                                                          B1DataType,
                                                                          D1sDataType,
                                                                          E1DataType,
                                                                          PassThrough,
                                                                          PassThrough,
                                                                          CDE0ElementOp,
                                                                          PassThrough,
                                                                          CDE1ElementOp>>
{
    using DeviceOp = DeviceBatchedGemmMultipleDGemmMultipleD<A0Layout,
                                                             B0Layout,
                                                             D0sLayout,
                                                             B1Layout,
                                                             D1sLayout,
                                                             E1Layout,
                                                             A0DataType,
                                                             B0DataType,
                                                             D0sDataType,
                                                             B1DataType,
                                                             D1sDataType,
                                                             E1DataType,
                                                             PassThrough,
                                                             PassThrough,
                                                             CDE0ElementOp,
                                                             PassThrough,
                                                             CDE1ElementOp>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<A0DataType, half_t> && is_same_v<B0DataType, half_t> &&
                     is_same_v<B1DataType, half_t> && is_same_v<E1DataType, half_t>)
        {
            if constexpr(is_same_v<A0Layout, Row> && is_same_v<B0Layout, Col> &&
                         is_same_v<B1Layout, Row> && is_same_v<E1Layout, Row>)
            {
                add_device_batched_gemm_add_relu_gemm_add_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance(
                    op_ptrs);
            }
            else if constexpr(is_same_v<A0Layout, Row> && is_same_v<B0Layout, Col> &&
                              is_same_v<B1Layout, Col> && is_same_v<E1Layout, Row>)
            {
                add_device_batched_gemm_add_relu_gemm_add_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gon_gmo_instance(
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
