// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Normalize                            = ck::tensor_operation::element_wise::Normalize;
using DeviceNormalizeFromMeanMeanSquarePtr = ck::tensor_operation::device::DeviceElementwiseBasePtr<
    Tuple<half_t, float, float, half_t, half_t>,
    Tuple<half_t>,
    Normalize,
    2>;

void add_device_normalize_from_mean_squaremean_f16_f32_f32_f16_f16_instances(
    std::vector<DeviceNormalizeFromMeanMeanSquarePtr>& instances);

template <typename InputType,
          typename MeanType,
          typename MeanSquareType,
          typename GammaDataType,
          typename BetaDataType,
          typename OutputType>
auto get_device_normalize_from_mean_meansquare_instances()
{
    std::vector<DeviceNormalizeFromMeanMeanSquarePtr> op_ptrs;

    if constexpr(is_same<InputType, half_t>::value && is_same<MeanType, float>::value &&
                 is_same<MeanSquareType, float>::value && is_same<GammaDataType, half_t>::value &&
                 is_same<BetaDataType, half_t>::value && is_same<OutputType, half_t>::value)
    {
        ck::tensor_operation::device::instance::
            add_device_normalize_from_mean_squaremean_f16_f32_f32_f16_f16_instances(op_ptrs);
    }

    return op_ptrs;
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
