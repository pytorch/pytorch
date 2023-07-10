// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batchnorm_forward_impl.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// clang-format off
template <index_t Rank, index_t NumReduceDim, typename YElementwiseOp>
using device_batchnorm_forward_f16_blockwise_instances =
     std::tuple <
        // XDataType, YDataType, AccDataType, ScaleDataType, BiasDataType, MeanVarDataType, YElementwiseOp, Rank, NumReduceDim, UseMultiBlockInK, BLockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XSrcYDstVectorDim, XSrcVectorSize, YDstVectorSize, ScaleSrcVectorSize, BiasSrcVectorSize, MeanVarSrcDstVectorSize 
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 128, 2,    2,    2,    0,    2,    2,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 128, 2,    2,    2,    0,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 128, 2,    2,    2,    0,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 128, 2,    2,    2,    0,    2,    2,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 128, 2,    2,    2,    1,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 128, 2,    2,    2,    1,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 64,  4,    2,    2,    0,    2,    2,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 64,  4,    2,    2,    0,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 64,  4,    2,    2,    0,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 64,  4,    2,    2,    0,    2,    2,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 64,  4,    2,    2,    1,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 64,  4,    2,    2,    1,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 32,  8,    2,    2,    0,    2,    2,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 32,  8,    2,    2,    0,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 32,  8,    2,    2,    0,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 32,  8,    2,    2,    0,    2,    2,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 32,  8,    2,    2,    1,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 32,  8,    2,    2,    1,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 16, 16,    2,    2,    0,    2,    2,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 16, 16,    2,    2,    0,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 16, 16,    2,    2,    0,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 16, 16,    2,    2,    0,    2,    2,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 16, 16,    2,    2,    1,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 16, 16,    2,    2,    1,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 8,  32,    2,    2,    0,    2,    2,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 8,  32,    2,    2,    0,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 8,  32,    2,    2,    0,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 8,  32,    2,    2,    0,    2,    2,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 8,  32,    2,    2,    1,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 8,  32,    2,    2,    1,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 4,  64,    2,    2,    0,    2,    2,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 4,  64,    2,    2,    0,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 4,  64,    2,    2,    0,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 4,  64,    2,    2,    0,    2,    2,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 4,  64,    2,    2,    1,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 4,  64,    2,    2,    1,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 2, 128,    2,    2,    0,    2,    2,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 2, 128,    2,    2,    0,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 2, 128,    2,    2,    0,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 2, 128,    2,    2,    0,    2,    2,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 2, 128,    2,    2,    1,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 2, 128,    2,    2,    1,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 1, 256,    2,    2,    0,    2,    2,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 1, 256,    2,    2,    0,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 1, 256,    2,    2,    0,    1,    1,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 1, 256,    2,    2,    0,    2,    2,    1,    1,    1>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 1, 256,    2,    2,    1,    1,    1,    2,    2,    2>,  
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, false, 256, 1, 256,    2,    2,    1,    1,    1,    1,    1,    1>
     >;
// clang-format on

// clang-format off
template <index_t Rank, index_t NumReduceDim, typename YElementwiseOp>
using device_batchnorm_forward_f16_multiblock_instances =
     std::tuple <
        // XDataType, YDataType, AccDataType, ScaleDataType, BiasDataType, MeanVarDataType, YElementwiseOp, Rank, NumReduceDim, UseMultiBlockInK, BLockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XSrcYDstVectorDim, XSrcVectorSize, YDstVectorSize, ScaleSrcVectorSize, BiasSrcVectorSize, MeanVarSrcDstVectorSize
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 128, 2,    2,    2,    0,    2,    2,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 128, 2,    2,    2,    0,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 128, 2,    2,    2,    0,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 128, 2,    2,    2,    0,    2,    2,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 128, 2,    2,    2,    1,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 128, 2,    2,    2,    1,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 64,  4,    2,    2,    0,    2,    2,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 64,  4,    2,    2,    0,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 64,  4,    2,    2,    0,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 64,  4,    2,    2,    0,    2,    2,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 64,  4,    2,    2,    1,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 64,  4,    2,    2,    1,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 32,  8,    2,    2,    0,    2,    2,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 32,  8,    2,    2,    0,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 32,  8,    2,    2,    0,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 32,  8,    2,    2,    0,    2,    2,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 32,  8,    2,    2,    1,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 32,  8,    2,    2,    1,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 16, 16,    2,    2,    0,    2,    2,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 16, 16,    2,    2,    0,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 16, 16,    2,    2,    0,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 16, 16,    2,    2,    0,    2,    2,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 16, 16,    2,    2,    1,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 16, 16,    2,    2,    1,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 8,  32,    2,    2,    0,    2,    2,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 8,  32,    2,    2,    0,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 8,  32,    2,    2,    0,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 8,  32,    2,    2,    0,    2,    2,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 8,  32,    2,    2,    1,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 8,  32,    2,    2,    1,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 4,  64,    2,    2,    0,    2,    2,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 4,  64,    2,    2,    0,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 4,  64,    2,    2,    0,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 4,  64,    2,    2,    0,    2,    2,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 4,  64,    2,    2,    1,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 4,  64,    2,    2,    1,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 2, 128,    2,    2,    0,    2,    2,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 2, 128,    2,    2,    0,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 2, 128,    2,    2,    0,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 2, 128,    2,    2,    0,    2,    2,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 2, 128,    2,    2,    1,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 2, 128,    2,    2,    1,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 1, 256,    2,    2,    0,    2,    2,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 1, 256,    2,    2,    0,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 1, 256,    2,    2,    0,    1,    1,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 1, 256,    2,    2,    0,    2,    2,    1,    1,    1>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 1, 256,    2,    2,    1,    1,    1,    2,    2,    2>,
        DeviceBatchNormFwdImpl<F16, F16, F32, F16, F16, F32, YElementwiseOp, Rank, NumReduceDim, true, 256, 1, 256,    2,    2,    1,    1,    1,    1,    1,    1>
     >;
// clang-format on

void add_device_batchnorm_forward_rank_4_3_f16_instances(
    std::vector<
        std::unique_ptr<DeviceBatchNormFwd<F16, F16, F32, F16, F16, F32, PassThrough, 4, 3>>>&
        instances)
{
    add_device_operation_instances(
        instances, device_batchnorm_forward_f16_blockwise_instances<4, 3, PassThrough>{});
    add_device_operation_instances(
        instances, device_batchnorm_forward_f16_multiblock_instances<4, 3, PassThrough>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
