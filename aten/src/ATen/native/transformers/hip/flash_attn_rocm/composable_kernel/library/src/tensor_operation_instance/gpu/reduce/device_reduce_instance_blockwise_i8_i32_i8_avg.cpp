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
template void add_device_reduce_instance_blockwise<I8, I32, I8, 4, 3, ReduceAdd, PassThrough, UnaryDivide, false, false>(std::vector<DeviceReducePtr<4, 3, PassThrough, UnaryDivide>>&); 
template void add_device_reduce_instance_blockwise<I8, I32, I8, 4, 4, ReduceAdd, PassThrough, UnaryDivide, false, false>(std::vector<DeviceReducePtr<4, 4, PassThrough, UnaryDivide>>&); 
template void add_device_reduce_instance_blockwise<I8, I32, I8, 4, 1, ReduceAdd, PassThrough, UnaryDivide, false, false>(std::vector<DeviceReducePtr<4, 1, PassThrough, UnaryDivide>>&); 
template void add_device_reduce_instance_blockwise<I8, I32, I8, 2, 1, ReduceAdd, PassThrough, UnaryDivide, false, false>(std::vector<DeviceReducePtr<2, 1, PassThrough, UnaryDivide>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation

} // namespace ck
