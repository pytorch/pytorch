// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename YDataType,
          typename AccElementwiseOperation,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceNormalization : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> lengths,
                        const std::vector<index_t> xStrides,
                        const std::vector<index_t> gammaStrides,
                        const std::vector<index_t> betaStrides,
                        const std::vector<index_t> yStrides,
                        const std::vector<index_t> reduceDims,
                        AccDataType epsilon,
                        const void* p_x,
                        const void* p_gamma,
                        const void* p_beta,
                        void* p_y,
                        void* p_savedMean,
                        void* p_savedInvVar,
                        AccElementwiseOperation acc_elementwise_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename YDataType,
          typename AccElementwiseOperation,
          index_t Rank,
          index_t NumReduceDim>
using DeviceNormalizationPtr = std::unique_ptr<DeviceNormalization<XDataType,
                                                                   GammaDataType,
                                                                   BetaDataType,
                                                                   AccDataType,
                                                                   YDataType,
                                                                   AccElementwiseOperation,
                                                                   Rank,
                                                                   NumReduceDim>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
