// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_reduce_multiblock.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_impl_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using reduce_configuration_1_instances_blockwise = std::tuple<
    // clang-format off
    // BlockSize | MThreadClusterSize | KThreadClusterSize
    ReductionConfiguration_1<256, 128, 2>,
    ReductionConfiguration_1<256, 64, 4>,
    ReductionConfiguration_1<256, 32, 8>,
    ReductionConfiguration_1<256, 16, 16>,
    ReductionConfiguration_1<256, 8, 32>,
    ReductionConfiguration_1<256, 4, 64>,
    ReductionConfiguration_1<256, 2, 128>,
    ReductionConfiguration_1<256, 1, 256>
    // clang-format on
    >;

#ifdef QUICK_REDUCE_TEST
using reduce_configuration_2_instances_blockwise = std::tuple<
    // clang-format off
    // InSrcVectorDim | InSrcVectorSize | OutDstVectorSize | MThreadSliceSize | KThreadSliceSize
    ReductionConfiguration_2<0, 2, 2, 2, 1>,
    ReductionConfiguration_2<0, 1, 1, 2, 1>,
    ReductionConfiguration_2<1, 2, 1, 1, 2>,
    ReductionConfiguration_2<0, 1, 1, 3, 1>,
    ReductionConfiguration_2<1, 1, 1, 1, 3>
    // clang-format on
    >;
#else
using reduce_configuration_2_instances_blockwise = std::tuple<
    // clang-format off
    // InSrcVectorDim | InSrcVectorSize | OutDstVectorSize | MThreadSliceSize | KThreadSliceSize
    ReductionConfiguration_2<0, 4, 4, 8, 1>,
    ReductionConfiguration_2<0, 4, 4, 4, 1>,
    ReductionConfiguration_2<0, 2, 2, 2, 1>,

    ReductionConfiguration_2<1, 4, 1, 1, 8>,
    ReductionConfiguration_2<1, 4, 1, 1, 4>,
    ReductionConfiguration_2<1, 2, 1, 1, 2>,

    // special instances
    ReductionConfiguration_2<0, 1, 1, 3, 1>,
    ReductionConfiguration_2<0, 1, 1, 5, 1>,
    ReductionConfiguration_2<0, 1, 1, 7, 1>,
    ReductionConfiguration_2<0, 1, 1, 11, 1>,

    ReductionConfiguration_2<1, 1, 1, 1, 3>,
    ReductionConfiguration_2<1, 1, 1, 1, 5>,
    ReductionConfiguration_2<1, 1, 1, 1, 7>,
    ReductionConfiguration_2<1, 1, 1, 1, 11>
    // clang-format on
    >;
#endif

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          int Rank,
          int NumReduceDim,
          typename ReduceOperation,
          typename InElementwiseOp,
          typename AccElementwiseOp,
          bool PropagateNan,
          bool OutputIndex>
void add_device_reduce_instance_blockwise(
    std::vector<DeviceReducePtr<Rank, NumReduceDim, InElementwiseOp, AccElementwiseOp>>&
        device_op_instances)
{
    static_for<0, std::tuple_size<reduce_configuration_1_instances_blockwise>::value, 1>{}(
        [&](auto i) {
            using cfg1 = remove_cvref_t<decltype(
                std::get<i.value>(reduce_configuration_1_instances_blockwise{}))>;

            static_for<0, std::tuple_size<reduce_configuration_2_instances_blockwise>::value, 1>{}(
                [&](auto j) {
                    using cfg2 = remove_cvref_t<decltype(
                        std::get<j.value>(reduce_configuration_2_instances_blockwise{}))>;

                    using ReduceOpInstance =
                        DeviceReduceMultiBlock<InDataType,
                                               AccDataType,
                                               OutDataType,
                                               Rank,
                                               NumReduceDim,
                                               ReduceOperation,
                                               InElementwiseOp,
                                               AccElementwiseOp,
                                               InMemoryDataOperationEnum::Set,
                                               PropagateNan,
                                               OutputIndex,
                                               false, // HaveIndexInputIfOutputIndex
                                               cfg1::BlockSize_,
                                               cfg1::MThreadClusterSize_,
                                               cfg1::KThreadClusterSize_,
                                               cfg2::MThreadSliceSize_,
                                               cfg2::KThreadSliceSize_,
                                               cfg2::InSrcVectorDim_,
                                               cfg2::InSrcVectorSize_,
                                               cfg2::OutDstVectorSize_>;

                    device_op_instances.push_back(
                        std::make_unique<ReduceOpInstance>(ReduceOpInstance{}));
                });
        });
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
