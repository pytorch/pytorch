// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_dl_f16_f16_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f16_f16_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f32_f32_f32_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&

        instances);

void add_device_gemm_dl_f32_f32_f32_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f32_f32_f32_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_f32_f32_f32_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_i8_i8_i8_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_i8_i8_i8_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_i8_i8_i8_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_dl_i8_i8_i8_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f32_f32_f32_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f32_f32_f32_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f32_f32_f32_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_f32_f32_f32_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_i8_i8_i8_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_i8_i8_i8_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_i8_i8_i8_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_c_shuffle_i8_i8_i8_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f16_f16_f16_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f16_f16_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f32_f32_f32_km_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f32_f32_f32_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f32_f32_f32_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f32_f32_f32_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F32, F32, F32, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f64_f64_f64_km_kn_mn_instances(
    std::vector<std::unique_ptr<

        DeviceGemm<Col, Row, Row, F64, F64, F64, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f64_f64_f64_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, F64, F64, F64, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f64_f64_f64_mk_kn_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, F64, F64, F64, PassThrough, PassThrough, PassThrough>>>&
        instances);

void add_device_gemm_xdl_f64_f64_f64_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Col, Row, F64, F64, F64, PassThrough, PassThrough, PassThrough>>>&
        instances);

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemm<ALayout,
                                             BLayout,
                                             CLayout,
                                             ADataType,
                                             BDataType,
                                             CDataType,
                                             ck::tensor_operation::element_wise::PassThrough,
                                             ck::tensor_operation::element_wise::PassThrough,
                                             ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceGemm<ALayout,
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
                add_device_gemm_xdl_f32_f32_f32_mk_kn_mn_instances(op_ptrs);
                add_device_gemm_dl_f32_f32_f32_mk_kn_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_f32_f32_f32_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_f32_f32_f32_mk_nk_mn_instances(op_ptrs);
                add_device_gemm_dl_f32_f32_f32_mk_nk_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_f32_f32_f32_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_f32_f32_f32_km_kn_mn_instances(op_ptrs);
                add_device_gemm_dl_f32_f32_f32_km_kn_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_f32_f32_f32_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_f32_f32_f32_km_nk_mn_instances(op_ptrs);
                add_device_gemm_dl_f32_f32_f32_km_nk_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_f32_f32_f32_km_nk_mn_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                          is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
                add_device_gemm_dl_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
                add_device_gemm_dl_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_f16_f16_f16_km_kn_mn_instances(op_ptrs);
                add_device_gemm_dl_f16_f16_f16_km_kn_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_f16_f16_f16_km_nk_mn_instances(op_ptrs);
                add_device_gemm_dl_f16_f16_f16_km_nk_mn_instances(op_ptrs);
                add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ADataType, ck::bhalf_t> && is_same_v<BDataType, ck::bhalf_t> &&
                          is_same_v<CDataType, ck::bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<ADataType, int8_t> && is_same_v<BDataType, int8_t> &&
                          is_same_v<CDataType, int8_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_i8_i8_i8_mk_kn_mn_instances(op_ptrs);
                add_device_gemm_dl_i8_i8_i8_mk_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_i8_i8_i8_mk_nk_mn_instances(op_ptrs);
                add_device_gemm_dl_i8_i8_i8_mk_nk_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Row> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_i8_i8_i8_km_kn_mn_instances(op_ptrs);
                add_device_gemm_dl_i8_i8_i8_km_kn_mn_instances(op_ptrs);
            }
            else if constexpr(is_same_v<ALayout, Col> && is_same_v<BLayout, Col> &&
                              is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_c_shuffle_i8_i8_i8_km_nk_mn_instances(op_ptrs);
                add_device_gemm_dl_i8_i8_i8_km_nk_mn_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
