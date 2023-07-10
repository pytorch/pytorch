// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          typename YElementwiseOp,
          index_t Rank,
          index_t NumBatchNormReduceDim>
struct DeviceBatchNormFwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const std::array<index_t, Rank> xyLengths,
        const std::array<index_t, Rank> xStrides,
        const std::array<index_t, Rank> yStrides,
        const std::array<int, NumBatchNormReduceDim> reduceDims,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarLengths,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleStrides,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnBiasStrides,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnMeanVarStrides,
        const void* p_x,
        const void* bnScale,
        const void* bnBias,
        double epsilon,
        const YElementwiseOp y_elementwise_op,
        void* p_y,
        void* resultSaveMean,
        void* resultSaveInvVariance,
        double exponentialAverageFactor,
        void* resultRunningMean,
        void* resultRunningVariance) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename BiasDataType,
          typename MeanVarDataType,
          typename YElementwiseOp,
          index_t Rank,
          index_t NumBatchNormReduceDim>
using DeviceBatchNormFwdPtr = std::unique_ptr<DeviceBatchNormFwd<XDataType,
                                                                 YDataType,
                                                                 AccDataType,
                                                                 ScaleDataType,
                                                                 BiasDataType,
                                                                 MeanVarDataType,
                                                                 YElementwiseOp,
                                                                 Rank,
                                                                 NumBatchNormReduceDim>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
