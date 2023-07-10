// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceBatchedGemmMultiD : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    static_assert(DsLayout::Size() == DsDataType::Size(), "wrong! inconsisiten NumDTensor");

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const std::array<const void*, NumDTensor>& p_ds,
                        void* p_e,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t Batch,
                        index_t StrideA,
                        index_t StrideB,
                        const std::array<ck::index_t, NumDTensor>& StrideDs,
                        index_t StrideE,
                        index_t BatchStrideA,
                        index_t BatchStrideB,
                        const std::array<ck::index_t, NumDTensor>& BatchStrideDs,
                        index_t BatchStrideE,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
