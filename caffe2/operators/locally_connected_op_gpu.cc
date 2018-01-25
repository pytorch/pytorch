/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/locally_connected_op.h"
#include "caffe2/operators/locally_connected_op_impl.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(LC, LocallyConnectedOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    LCGradient,
    LocallyConnectedGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(LC1D, LocallyConnectedOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    LC1DGradient,
    LocallyConnectedGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(LC2D, LocallyConnectedOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    LC2DGradient,
    LocallyConnectedGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(LC3D, LocallyConnectedOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    LC3DGradient,
    LocallyConnectedGradientOp<float, CUDAContext>);

} // namespace caffe2
