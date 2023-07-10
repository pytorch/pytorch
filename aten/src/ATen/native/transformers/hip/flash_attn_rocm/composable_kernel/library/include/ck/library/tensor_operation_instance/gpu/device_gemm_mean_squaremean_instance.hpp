// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_reduce.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using DeviceGemmAddAddMeanSquareMeanPtr = ck::tensor_operation::device::DeviceGemmReducePtr<1, 2>;

void add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_mk_kn_mn_instances(
    std::vector<DeviceGemmAddAddMeanSquareMeanPtr>&);
void add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_mk_nk_mn_instances(
    std::vector<DeviceGemmAddAddMeanSquareMeanPtr>&);
void add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_km_kn_mn_instances(
    std::vector<DeviceGemmAddAddMeanSquareMeanPtr>&);
void add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_km_nk_mn_instances(
    std::vector<DeviceGemmAddAddMeanSquareMeanPtr>&);

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
auto get_device_gemm_add_add_mean_squaremean_instances()
{
    std::vector<DeviceGemmAddAddMeanSquareMeanPtr> op_ptrs;

    if constexpr(is_same<ADataType, half_t>::value && is_same<BDataType, half_t>::value &&
                 is_same<CDataType, half_t>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_mk_kn_mn_instances(
                    op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_mk_nk_mn_instances(
                    op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_km_kn_mn_instances(
                    op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_km_nk_mn_instances(
                    op_ptrs);
        }
    }

    return op_ptrs;
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
