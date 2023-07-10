// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// conv2d backward data
void add_device_grouped_conv2d_bwd_data_xdl_gnhwc_gkyxc_gnhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

template <ck::index_t NumDimSpatial,
          typename OutLayout,
          typename WeiLayout,
          typename InLayout,
          typename OutDataType,
          typename WeiDataType,
          typename InDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<
        NumDimSpatial,
        OutLayout,
        WeiLayout,
        Empty_Tuple,
        InLayout,
        OutDataType,
        WeiDataType,
        Empty_Tuple,
        InDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp =
        DeviceGroupedConvBwdDataMultipleD<NumDimSpatial,
                                          OutLayout,
                                          WeiLayout,
                                          Empty_Tuple,
                                          InLayout,
                                          OutDataType,
                                          WeiDataType,
                                          Empty_Tuple,
                                          InDataType,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, GNHWC> &&
                     is_same_v<WeiLayout, GKYXC> && is_same_v<OutLayout, GNHWK>)
        {
            if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                         is_same_v<OutDataType, F16>)
            {
                add_device_grouped_conv2d_bwd_data_xdl_gnhwc_gkyxc_gnhwk_f16_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
