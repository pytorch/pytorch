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
#include "caffe2/operators/stump_func_op.h"

namespace caffe2 {

namespace {

template <typename TIN, typename TOUT>
__global__ void StumpFuncKernel(
  const int N,
  const TIN threshold,
  const TOUT low_value,
  const TOUT high_value,
  const TIN* X,
  TOUT* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = (X[i] <= threshold) ? low_value : high_value;
  }
}

} //

template <>
bool StumpFuncOp<float, float, CUDAContext>::RunOnDevice() {
  auto& in = Input(0);
  const float* in_data = in.data<float>();

  auto* out = Output(0, in.sizes(), at::dtype<float>());
  float* out_data = out->template mutable_data<float>();
  StumpFuncKernel<<<CAFFE_GET_BLOCKS(in.numel()), CAFFE_CUDA_NUM_THREADS,
    0, context_.cuda_stream()>>>(
      in.numel(), threshold_, low_value_, high_value_, in_data, out_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(StumpFunc, StumpFuncOp<float, float, CUDAContext>);
// NO_GRADIENT(StumpFuncGpu);

} // caffe2
