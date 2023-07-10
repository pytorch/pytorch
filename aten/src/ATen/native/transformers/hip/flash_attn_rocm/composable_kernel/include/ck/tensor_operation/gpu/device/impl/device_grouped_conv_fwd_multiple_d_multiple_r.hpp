// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Grouped Convolution Forward:
//   input : input image A[G, N, C, Hi, Wi],
//   input : weight B[G, K, C, Y, X],
//   input : D0[G, N, K, Ho, Wo], D1[G, N, K, Ho, Wo], ...
//   output : output image E[G, N, K, Ho, Wo]
//   output : R0[G, N, Ho, Wo], R1[G, N, Ho, Wo], ...
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
//   Q0 = reduce0(q_op0(E)), Q1 = reduce1(q_op0(E)), ...
//   R0 = r_op0(Q0), R1 = r_op1(Q1), ...
// Assume:
//   D0, D1, ... and E have the same layout
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DELayout,
          typename RLayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename RsDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename QsElementwiseOperation,
          typename RsElementwiseOperation>
struct DeviceGroupedConvFwdMultipleDMultipleR : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();
    static constexpr index_t NumRTensor = RsDataType::Size();

    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b,
        const std::array<const void*, NumDTensor>& p_ds,
        void* p_e,
        std::array<void*, NumRTensor> p_rs,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 2>& r_g_n_wos_lengths,
        const std::array<index_t, NDimSpatial + 2>& r_g_n_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CDEElementwiseOperation& cde_element_op,
        const QsElementwiseOperation& qs_element_op,
        const RsElementwiseOperation& rs_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
