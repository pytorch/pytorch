// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance_multiblock_atomic_add.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
// InDataType | AccDataType | OutDataType | Rank | NumReduceDim | ReduceOperation | InElementwiseOp | AccElementwiseOp | PropagateNan | UseIndex 
extern template void add_device_reduce_instance_multiblock_atomic_add<F64, F64, F64, 4, 3, ReduceAdd, PassThrough, UnaryDivide, false, false>(std::vector<DeviceReducePtr<4, 3, PassThrough, UnaryDivide>>&); 
extern template void add_device_reduce_instance_multiblock_atomic_add<F64, F64, F64, 4, 4, ReduceAdd, PassThrough, UnaryDivide, false, false>(std::vector<DeviceReducePtr<4, 4, PassThrough, UnaryDivide>>&); 
extern template void add_device_reduce_instance_multiblock_atomic_add<F64, F64, F64, 4, 1, ReduceAdd, PassThrough, UnaryDivide, false, false>(std::vector<DeviceReducePtr<4, 1, PassThrough, UnaryDivide>>&); 
extern template void add_device_reduce_instance_multiblock_atomic_add<F64, F64, F64, 2, 1, ReduceAdd, PassThrough, UnaryDivide, false, false>(std::vector<DeviceReducePtr<2, 1, PassThrough, UnaryDivide>>&);
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
