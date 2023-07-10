// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           F32_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Bilinear>>>& instances);

void add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_knnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           F32_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Bilinear>>>& instances);

void add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mknn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           F32_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Bilinear>>>& instances);

void add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mnnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           F32_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Bilinear>>>& instances);

// Contraction + Bilinear
template <index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename DDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceContractionMultipleD<
    NumDimM,
    NumDimN,
    NumDimK,
    ADataType,
    BDataType,
    ck::Tuple<DDataType>,
    EDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::Bilinear>>
{
    using DeviceOp = DeviceContractionMultipleD<NumDimM,
                                                NumDimN,
                                                NumDimK,
                                                ADataType,
                                                BDataType,
                                                ck::Tuple<DDataType>,
                                                EDataType,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::Bilinear>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float> &&
                     is_same_v<DDataType, float> && is_same_v<EDataType, float>)
        {
            if constexpr(NumDimM == 2 && NumDimN == 2 && NumDimK == 2)
            {
                add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_knnn_instance(
                    op_ptrs);
                add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_kknn_instance(
                    op_ptrs);
                add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mnnn_instance(
                    op_ptrs);
                add_device_contraction_bilinear_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_f32_mknn_instance(
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
