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

/* SampleAs by Kaiming He for Mask R-CNN
X.dim32(0) = L.dim32(0)
Y's output samples are the samples of X for which L > 0.
*/
#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "modules/detectron/sample_as_op.h"

#include <stdio.h>

namespace caffe2 {

template <>
bool SampleAsOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0); // Input data to be sliced
  auto& L = Input(1); // Target data that provide the identity

  CAFFE_ENFORCE(
      X.dim32(0) == L.dim32(0),
      "X.dim32(0) must be equal to L.dim32(0)",
      "(",
      X.dim32(0),
      " vs. ",
      L.dim32(0),
      ")");

  // copy L to CPU:
  std::vector<int> labels(L.dim32(0));
  context_.CopyBytes<CUDAContext, CPUContext>(
      L.dim32(0) * sizeof(int), L.data<int>(), &labels[0]);
  // Make sure that the copy is finished
  context_.FinishDeviceComputation();

  int count = 0;
  for (int i = 0; i < L.dim32(0); i++) {
    if (labels[i] > 0) {
      count++;
    }
  }
  assert(count > 0);

  // resize Y
  vector<int64_t> out_shape(X.sizes().vec());
  out_shape[0] = count;
  auto* Y = Output(0, out_shape, at::dtype<float>()); // Sliced data (Y.dim32(0) = num of (L > 0))

  const int len = X.size() / X.dim32(0);

  float* output = Y->mutable_data<float>();
  for (int i = 0; i < L.dim32(0); i++) {
    if (labels[i] > 0) {
      context_.CopyBytes<CUDAContext, CUDAContext>(
          len * sizeof(float), X.data<float>() + i * len, output);
      output += len;
    } // if
  } // i

  return true;
}

template <>
bool SampleAsGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& L = Input(1);
  auto& dY = Input(2);


  auto* dX = Output(0, X.sizes(), at::dtype<float>());

  // copy L to CPU:
  std::vector<int> labels(L.dim32(0));
  context_.CopyBytes<CUDAContext, CPUContext>(
      L.dim32(0) * sizeof(int), L.data<int>(), &labels[0]);
  // Make sure that the copy is finished
  context_.FinishDeviceComputation();

  // zero-out dX
  math::Set<float, CUDAContext>(
      dX->size(), 0.f, dX->mutable_data<float>(), &context_);

  const int len = X.size() / X.dim32(0);

  const float* input = dY.data<float>();
  for (int i = 0; i < L.dim32(0); i++) {
    if (labels[i] > 0) {
      context_.CopyBytes<CUDAContext, CUDAContext>(
          len * sizeof(float), input, dX->mutable_data<float>() + i * len);
      input += len;
    } // if
  } // i

  return true;
}

REGISTER_CUDA_OPERATOR(SampleAs, SampleAsOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SampleAsGradient,
    SampleAsGradientOp<float, CUDAContext>);
} // namespace caffe2
