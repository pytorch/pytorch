// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOperation | InElementwiseOp | AccElementwiseOp | PropagateNan | UseIndex 
template void add_device_reduce_instance_blockwise<F16, F16, F16, 4, 3, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<4, 3, PassThrough, PassThrough>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 4, 4, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<4, 4, PassThrough, PassThrough>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 4, 1, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<4, 1, PassThrough, PassThrough>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 2, 1, ReduceMin, PassThrough, PassThrough, false, false>(std::vector<DeviceReducePtr<2, 1, PassThrough, PassThrough>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 4, 3, ReduceMin, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<4, 3, PassThrough, PassThrough>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 4, 4, ReduceMin, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<4, 4, PassThrough, PassThrough>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 4, 1, ReduceMin, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<4, 1, PassThrough, PassThrough>>&);
template void add_device_reduce_instance_blockwise<F16, F16, F16, 2, 1, ReduceMin, PassThrough, PassThrough, false, true>(std::vector<DeviceReducePtr<2, 1, PassThrough, PassThrough>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
