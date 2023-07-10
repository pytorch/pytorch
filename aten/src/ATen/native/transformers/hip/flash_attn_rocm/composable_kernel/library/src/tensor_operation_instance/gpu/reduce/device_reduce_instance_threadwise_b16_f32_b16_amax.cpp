// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/reduction_enums.hpp"
#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_threadwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOperation | InElementwiseOp | AccElementwiseOp | PropagateNan | UseIndex 
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<4, 3, UnaryAbs, PassThrough>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<4, 4, UnaryAbs, PassThrough>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<4, 1, UnaryAbs, PassThrough>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, false>(std::vector<DeviceReducePtr<2, 1, UnaryAbs, PassThrough>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 4, 3, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<4, 3, UnaryAbs, PassThrough>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 4, 4, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<4, 4, UnaryAbs, PassThrough>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 4, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<4, 1, UnaryAbs, PassThrough>>&);
template void add_device_reduce_instance_threadwise<BF16, F32, BF16, 2, 1, ReduceAMax, UnaryAbs, PassThrough, false, true>(std::vector<DeviceReducePtr<2, 1, UnaryAbs, PassThrough>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
