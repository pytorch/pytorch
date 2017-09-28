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

#include "caffe2/operators/elementwise_op_test.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/flags.h"

CAFFE2_DECLARE_string(caffe_test_root);

template <>
void CopyVector<caffe2::CUDAContext>(const int N, const bool* x, bool* y) {
  cudaMemcpy(y, x, N * sizeof(bool), cudaMemcpyHostToDevice);
}

template <>
caffe2::OperatorDef CreateOperatorDef<caffe2::CUDAContext>() {
  caffe2::OperatorDef def;
  def.mutable_device_option()->set_device_type(caffe2::CUDA);
  return def;
}

TEST(ElementwiseGPUTest, And) {
  if (!caffe2::HasCudaGPU())
    return;
  elementwiseAnd<caffe2::CUDAContext>();
}

TEST(ElementwiseGPUTest, Or) {
  if (!caffe2::HasCudaGPU())
    return;
  elementwiseOr<caffe2::CUDAContext>();
}

TEST(ElementwiseGPUTest, Xor) {
  if (!caffe2::HasCudaGPU())
    return;
  elementwiseXor<caffe2::CUDAContext>();
}

TEST(ElementwiseGPUTest, Not) {
  if (!caffe2::HasCudaGPU())
    return;
  elementwiseNot<caffe2::CUDAContext>();
}
