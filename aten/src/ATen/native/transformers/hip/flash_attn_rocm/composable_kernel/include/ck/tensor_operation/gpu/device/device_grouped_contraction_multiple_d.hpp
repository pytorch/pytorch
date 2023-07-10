// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NumDTensor>
struct ContractionDesc
{
    std::vector<index_t> a_ms_ks_lengths;
    std::vector<index_t> a_ms_ks_strides;

    std::vector<index_t> b_ns_ks_lengths;
    std::vector<index_t> b_ns_ks_strides;

    std::array<std::vector<index_t>, NumDTensor> ds_ms_ns_lengths;
    std::array<std::vector<index_t>, NumDTensor> ds_ms_ns_strides;

    std::vector<index_t> e_ms_ns_lengths;
    std::vector<index_t> e_ms_ns_strides;
};

// Tensor Contraction:
//   input : A
//   input : B
//   input : D0, D1, ...
//   output : E
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   A[M0, M1, M2, ..., K0, K1, K2, ...]
//   B[N0, N1, N2, ..., K0, K1, K2, ...]
//   D[M0, M1, M2, ..., N0, N1, N2, ...]
//   E[M0, M1, M2, ..., N0, N1, N2, ...]
template <index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGroupedContractionMultipleD : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*> p_a_vec,
                        std::vector<const void*> p_b_vec,
                        std::vector<std::array<const void*, NumDTensor>> p_ds_vec,
                        std::vector<void*> p_e_vec,
                        std::vector<ContractionDesc<NumDTensor>> contraction_descs,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
