// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gkn_gmn_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemm<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gnk_gmn_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemm<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gkn_gmn_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemm<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gnk_gmn_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemm<Row, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemm<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_batched_gemm_xdl_f32_f32_f32_gkm_gkn_gmn_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemm<Col, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_batched_gemm_xdl_f32_f32_f32_gkm_gnk_gmn_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemm<Col, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_batched_gemm_xdl_f32_f32_f32_gmk_gkn_gmn_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemm<Row, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_batched_gemm_xdl_f32_f32_f32_gmk_gnk_gmn_instances(
    std::vector<std::unique_ptr<
        DeviceBatchedGemm<Row, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_batched_gemm_xdl_int8_int8_int8_gkm_gkn_gmn_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemm<Col,
                                                  Row,
                                                  Row,
                                                  int8_t,
                                                  int8_t,
                                                  int8_t,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_batched_gemm_xdl_int8_int8_int8_gkm_gnk_gmn_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemm<Col,
                                                  Col,
                                                  Row,
                                                  int8_t,
                                                  int8_t,
                                                  int8_t,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_batched_gemm_xdl_int8_int8_int8_gmk_gkn_gmn_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemm<Row,
                                                  Row,
                                                  Row,
                                                  int8_t,
                                                  int8_t,
                                                  int8_t,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

void add_device_batched_gemm_xdl_int8_int8_int8_gmk_gnk_gmn_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemm<Row,
                                                  Col,
                                                  Row,
                                                  int8_t,
                                                  int8_t,
                                                  int8_t,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances);

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceBatchedGemm<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceBatchedGemm<ALayout,
                                       BLayout,
                                       CLayout,
                                       ADataType,
                                       BDataType,
                                       CDataType,
                                       ck::tensor_operation::element_wise::PassThrough,
                                       ck::tensor_operation::element_wise::PassThrough,
                                       ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float> &&
                     is_same_v<CDataType, float>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_f32_f32_f32_gmk_gkn_gmn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_f32_f32_f32_gmk_gnk_gmn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_f32_f32_f32_gkm_gkn_gmn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_f32_f32_f32_gkm_gnk_gmn_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                          is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ADataType, bhalf_t> && is_same_v<BDataType, bhalf_t> &&
                          is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gkn_gmn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gnk_gmn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gkn_gmn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gnk_gmn_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ADataType, int8_t> && is_same_v<BDataType, int8_t> &&
                          is_same_v<CDataType, int8_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_int8_int8_int8_gmk_gkn_gmn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_int8_int8_int8_gmk_gnk_gmn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_int8_int8_int8_gkm_gkn_gmn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_batched_gemm_xdl_int8_int8_int8_gkm_gnk_gmn_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
