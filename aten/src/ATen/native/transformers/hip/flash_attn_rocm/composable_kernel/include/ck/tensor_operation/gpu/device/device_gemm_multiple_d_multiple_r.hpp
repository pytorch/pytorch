// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// FIXME: DeviceGemmReduce type need to well define the problem
// GEMM:
//   input : A[AK0, M, AK1]
//   input : B[AK0, N, AK1]
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   output : R0[M], R1[M], ...
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
//   Q0 = reduce0(q_op0(E)), Q1 = reduce1(q_op0(E)), ...
//   R0 = r_op0(Q0), R1 = r_op1(Q1), ...
// Assume:
//   D0, D1, ... and E have the same layout
template <typename ALayout,
          typename BLayout,
          typename DELayout,
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
struct DeviceGemmMultipleDMultipleR : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();
    static constexpr index_t NumRTensor = RsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        std::array<void*, NumRTensor> p_rs,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        ck::index_t StrideE,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op,
                        QsElementwiseOperation qs_element_op,
                        RsElementwiseOperation rs_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename ALayout,
          typename BLayout,
          typename DELayout,
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
using DeviceGemmMultipleDMultipleRPtr =
    std::unique_ptr<DeviceGemmMultipleDMultipleR<ALayout,
                                                 BLayout,
                                                 DELayout,
                                                 ADataType,
                                                 BDataType,
                                                 DsDataType,
                                                 EDataType,
                                                 RsDataType,
                                                 AElementwiseOperation,
                                                 BElementwiseOperation,
                                                 CDEElementwiseOperation,
                                                 QsElementwiseOperation,
                                                 RsElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
