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

#include "caffe2/operators/batch_matmul_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

template <>
bool BatchMatMulOp<CUDAContext, DefaultEngine>::RunOnDevice() {
    return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
}

REGISTER_CUDA_OPERATOR(BatchMatMul, BatchMatMulOp<CUDAContext>);

#if CUDA_VERSION >= 9000

template <>
bool BatchMatMulOp<CUDAContext, TensorCoreEngine>::RunOnDevice() {
    return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
}

REGISTER_CUDA_OPERATOR_WITH_ENGINE(
    BatchMatMul,
    TENSORCORE,
    BatchMatMulOp<CUDAContext, TensorCoreEngine>);
#endif

} // namespace caffe2
