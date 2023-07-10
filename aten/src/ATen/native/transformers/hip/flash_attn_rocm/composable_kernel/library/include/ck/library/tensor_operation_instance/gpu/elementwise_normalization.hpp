// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise_normalization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// FP16
void add_device_elementwise_normalization_rank_2_1_f16_instances(
    std::vector<std::unique_ptr<DeviceElementwiseNormalization<ck::Tuple<F16, F16>,
                                                               F16,
                                                               F16,
                                                               F32,
                                                               F16,
                                                               element_wise::Add,
                                                               PassThrough,
                                                               2,
                                                               1>>>&);

template <typename InDataTypeTuple,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceElementwiseNormalization<
    InDataTypeTuple,
    GammaDataType,
    BetaDataType,
    F32,
    YDataType,
    ck::tensor_operation::element_wise::Add,
    ck::tensor_operation::element_wise::PassThrough,
    Rank,
    NumReduceDim>>
{
    using DeviceOp = DeviceElementwiseNormalization<InDataTypeTuple,
                                                    GammaDataType,
                                                    BetaDataType,
                                                    F32,
                                                    YDataType,
                                                    ck::tensor_operation::element_wise::Add,
                                                    ck::tensor_operation::element_wise::PassThrough,
                                                    Rank,
                                                    NumReduceDim>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<GammaDataType, F16> && is_same_v<BetaDataType, F16> &&
                     is_same_v<YDataType, F16>)
        {
            if constexpr(Rank == 2 && NumReduceDim == 1)
            {
                add_device_elementwise_normalization_rank_2_1_f16_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
