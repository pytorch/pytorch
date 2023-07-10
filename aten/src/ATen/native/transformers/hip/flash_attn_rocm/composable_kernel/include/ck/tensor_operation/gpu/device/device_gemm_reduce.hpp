// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// FIXME: DeviceGemmReduce type need to well define the problem
template <ck::index_t NumDTensor, ck::index_t NumReduce>
struct DeviceGemmReduce : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const void* p_bias,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_c,
                        std::array<void*, NumReduce> p_reduces,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        std::array<void*, 3> gemm_element_ops,
                        std::array<void*, NumDTensor> d_element_ops,
                        std::array<void*, NumReduce> reduce_in_element_ops,
                        std::array<void*, NumReduce> reduce_out_element_ops,
                        ck::index_t BatchCount = 1) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <ck::index_t NumDTensor, ck::index_t NumReduce>
using DeviceGemmReducePtr = std::unique_ptr<DeviceGemmReduce<NumDTensor, NumReduce>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
