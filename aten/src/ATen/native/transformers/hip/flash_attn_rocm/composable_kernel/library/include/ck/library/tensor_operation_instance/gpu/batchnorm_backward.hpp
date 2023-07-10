// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_batchnorm_backward.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// FP16
void add_device_batchnorm_backward_rank_4_3_f16_instances(
    std::vector<std::unique_ptr<
        DeviceBatchNormBwd<F16, F32, F32, F32, F16, F32, F32, PassThrough, 4, 3>>>&);

// FP32
void add_device_batchnorm_backward_rank_4_3_f32_instances(
    std::vector<std::unique_ptr<
        DeviceBatchNormBwd<F32, F32, F32, F32, F32, F32, F32, PassThrough, 4, 3>>>&);

// BF16
void add_device_batchnorm_backward_rank_4_3_bf16_instances(
    std::vector<std::unique_ptr<
        DeviceBatchNormBwd<BF16, F32, F32, F32, BF16, F32, F32, PassThrough, 4, 3>>>&);

// FP64
void add_device_batchnorm_backward_rank_4_3_f64_instances(
    std::vector<std::unique_ptr<
        DeviceBatchNormBwd<F64, F64, F64, F64, F64, F64, F64, PassThrough, 4, 3>>>&);

template <typename XDataType,
          typename DxDataType,
          typename DyDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename DscaleDbiasDataType,
          typename MeanVarDataType,
          typename DyElementwiseOp,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceBatchNormBwd<XDataType,
                                                     DxDataType,
                                                     DyDataType,
                                                     AccDataType,
                                                     ScaleDataType,
                                                     DscaleDbiasDataType,
                                                     MeanVarDataType,
                                                     DyElementwiseOp,
                                                     Rank,
                                                     NumReduceDim>>
{
    using DeviceOp = DeviceBatchNormBwd<XDataType,
                                        DxDataType,
                                        DyDataType,
                                        AccDataType,
                                        ScaleDataType,
                                        DscaleDbiasDataType,
                                        MeanVarDataType,
                                        DyElementwiseOp,
                                        Rank,
                                        NumReduceDim>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<XDataType, F16> && is_same_v<DxDataType, F32> &&
                     is_same_v<DyDataType, F32> && is_same_v<AccDataType, F32> &&
                     is_same_v<ScaleDataType, F16> && is_same_v<DscaleDbiasDataType, F32> &&
                     is_same_v<MeanVarDataType, F32>)
        {
            if constexpr(Rank == 4 && NumReduceDim == 3 && is_same_v<DyElementwiseOp, PassThrough>)
            {
                add_device_batchnorm_backward_rank_4_3_f16_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<XDataType, F32> && is_same_v<DxDataType, F32> &&
                          is_same_v<DyDataType, F32> && is_same_v<AccDataType, F32> &&
                          is_same_v<ScaleDataType, F32> && is_same_v<DscaleDbiasDataType, F32> &&
                          is_same_v<MeanVarDataType, F32>)
        {
            if constexpr(Rank == 4 && NumReduceDim == 3 && is_same_v<DyElementwiseOp, PassThrough>)
            {
                add_device_batchnorm_backward_rank_4_3_f32_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<XDataType, BF16> && is_same_v<DxDataType, F32> &&
                          is_same_v<DyDataType, F32> && is_same_v<AccDataType, F32> &&
                          is_same_v<ScaleDataType, BF16> && is_same_v<DscaleDbiasDataType, F32> &&
                          is_same_v<MeanVarDataType, F32>)
        {
            if constexpr(Rank == 4 && NumReduceDim == 3 && is_same_v<DyElementwiseOp, PassThrough>)
            {
                add_device_batchnorm_backward_rank_4_3_bf16_instances(op_ptrs);
            }
        }
        else if constexpr(is_same_v<XDataType, F64> && is_same_v<DxDataType, F64> &&
                          is_same_v<DyDataType, F64> && is_same_v<AccDataType, F64> &&
                          is_same_v<ScaleDataType, F64> && is_same_v<DscaleDbiasDataType, F64> &&
                          is_same_v<MeanVarDataType, F64>)
        {
            if constexpr(Rank == 4 && NumReduceDim == 3 && is_same_v<DyElementwiseOp, PassThrough>)
            {
                add_device_batchnorm_backward_rank_4_3_f64_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
