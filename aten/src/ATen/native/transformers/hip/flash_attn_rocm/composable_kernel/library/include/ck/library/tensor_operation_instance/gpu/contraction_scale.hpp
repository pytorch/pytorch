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

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_knn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale>>>& instances);

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mnn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           F32,
                                                           F32,
                                                           Empty_Tuple,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale>>>& instances);

// Contraction + Scale
template <index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceContractionMultipleD<
    NumDimM,
    NumDimN,
    NumDimK,
    ADataType,
    BDataType,
    ck::Tuple<>,
    EDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::Scale>>
{
    using DeviceOp = DeviceContractionMultipleD<NumDimM,
                                                NumDimN,
                                                NumDimK,
                                                ADataType,
                                                BDataType,
                                                ck::Tuple<>,
                                                EDataType,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::Scale>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<ADataType, float> && is_same_v<BDataType, float> &&
                     is_same_v<EDataType, float>)
        {
            if constexpr(NumDimM == 2 && NumDimN == 2 && NumDimK == 2)
            {
                add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_kkn_instance(
                    op_ptrs);
                add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_knn_instance(
                    op_ptrs);
                add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mkn_instance(
                    op_ptrs);
                add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_f32_f32_f32_mnn_instance(
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
