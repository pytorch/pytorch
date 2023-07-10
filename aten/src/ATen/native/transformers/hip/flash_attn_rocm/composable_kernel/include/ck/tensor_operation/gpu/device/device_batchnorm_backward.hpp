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
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          typename DyElementwiseOp,
          index_t Rank,
          index_t NumBatchNormReduceDim>
struct DeviceBatchNormBwd : public BaseOperator
{
    static constexpr index_t NumInvariantDim = Rank - NumBatchNormReduceDim;

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, Rank> xyLengths,
                        const std::array<index_t, Rank> xStrides,
                        const std::array<index_t, Rank> dyStrides,
                        const std::array<index_t, Rank> dxStrides,
                        const std::array<int, NumBatchNormReduceDim> reduceDims,
                        const std::array<ck::index_t, NumInvariantDim> bnScaleBiasMeanVarLengths,
                        const std::array<ck::index_t, NumInvariantDim> bnScaleStrides,
                        const std::array<ck::index_t, NumInvariantDim> bnDscaleDbiasStrides,
                        const std::array<ck::index_t, NumInvariantDim> bnMeanVarStrides,
                        const void* p_x,
                        const void* p_dy,
                        const void* p_scale,
                        const void* p_savedMean,
                        const void* p_savedInvVar,
                        double epsilon,
                        const DyElementwiseOp dy_elementwise_op,
                        void* p_dx,
                        void* p_dscale,
                        void* p_dbias) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          typename DyElementwiseOp,
          index_t Rank,
          index_t NumBatchNormReduceDim>
using DeviceBatchNormBwdPtr = std::unique_ptr<DeviceBatchNormBwd<XDataType,
                                                                 DxDataType,
                                                                 DyDataType,
                                                                 AccDataType,
                                                                 ScaleDataType,
                                                                 DscaleDbiasDataType,
                                                                 MeanVarDataType,
                                                                 DyElementwiseOp,
                                                                 Rank,
                                                                 NumBatchNormReduceDim>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
