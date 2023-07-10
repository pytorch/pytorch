// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Tensor Contraction:
//   input : A
//   input : B
//   input : D0, D1, ...
//   output : E
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   A[G0, G1, ..., M0, M1, M2, ..., K0, K1, K2, ...]
//   B[G0, G1, ..., N0, N1, N2, ..., K0, K1, K2, ...]
//   D[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2, ...]
//   E[G0, G1, ..., M0, M1, M2, ..., N0, N1, N2, ...]
template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceSplitKContractionMultipleD : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        const std::vector<index_t>& a_gs_ms_ns_lengths,
                        const std::vector<index_t>& a_gs_ms_ks_strides,
                        const std::vector<index_t>& b_gs_ns_ks_lengths,
                        const std::vector<index_t>& b_gs_ns_ks_strides,
                        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_lengths,
                        const std::array<std::vector<index_t>, NumDTensor>& ds_gs_ms_ns_strides,
                        const std::vector<index_t>& e_gs_ms_ns_lengths,
                        const std::vector<index_t>& e_gs_ms_ns_strides,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op,
                        index_t split_k) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
