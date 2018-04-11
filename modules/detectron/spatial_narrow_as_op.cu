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
#include "caffe2/core/operator.h"
#include "spatial_narrow_as_op.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void CopyKernel(
    const int N,
    const int C,
    const int in_H,
    const int in_W,
    const int out_H,
    const int out_W,
    const T* in_data,
    T* out_data) {
  CUDA_1D_KERNEL_LOOP(index, N * C * out_H * out_W) {
    int w = index % out_W;
    int h = (index / out_W) % out_H;
    int c = (index / out_W / out_H) % C;
    int n = (index / out_W / out_H / C);
    int in_index = n * C * in_H * in_W + c * in_H * in_W + h * in_W + w;
    int out_index = n * C * out_H * out_W + c * out_H * out_W + h * out_W + w;
    out_data[out_index] = in_data[in_index];
  }
}

template <typename T>
__global__ void CopyGradientKernel(
    const int N,
    const int C,
    const int in_H,
    const int in_W,
    const int out_H,
    const int out_W,
    const T* in_data,
    T* out_data) {
  CUDA_1D_KERNEL_LOOP(index, N * C * in_H * in_W) {
    int w = index % in_W;
    int h = (index / in_W) % in_H;
    int c = (index / in_W / in_H) % C;
    int n = (index / in_W / in_H / C);
    int in_index = n * C * in_H * in_W + c * in_H * in_W + h * in_W + w;
    int out_index = n * C * out_H * out_W + c * out_H * out_W + h * out_W + w;
    out_data[out_index] = in_data[in_index];
  }
}
} // namespace


template <>
bool SpatialNarrowAsOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float_t, int32_t>>::call(this, Input(0));
}

template <>
template <typename T>
bool SpatialNarrowAsOp<CUDAContext>::DoRunWithType() {
  // Narrows input 0 (A) spatially to match input 1 (B)
  auto& A = Input(0);
  auto& B = Input(1);
  auto* C = Output(0);

  CAFFE_ENFORCE_EQ(A.dim32(0), B.dim32(0), "Input dim 0 must be equal.");
  if (A.ndim() == B.ndim()) {
    CAFFE_ENFORCE_EQ(A.dim32(1), B.dim32(1), "Input dim 1 must be equal.");
    CAFFE_ENFORCE_GE(
        A.dim32(2), B.dim32(2), "Input 0 height must be >= input 1 height.");
    CAFFE_ENFORCE_GE(
        A.dim32(3), B.dim32(3), "Input 0 width must be >= input 1 width.");

    C->ResizeLike(B);
  } else {
    // For (N, H, W) case
    CAFFE_ENFORCE_EQ(A.ndim() - 1, B.ndim(), "Dimension mismatch.");
    CAFFE_ENFORCE_GE(
        A.dim32(2), B.dim32(1), "Input 0 height must be >= input 1 height.");
    CAFFE_ENFORCE_GE(
        A.dim32(3), B.dim32(2), "Input 0 width must be >= input 1 width.");
    C->Resize(A.dim32(0), A.dim32(1), B.dim32(1), B.dim32(2));
  }
  int out_width = C->dim32(3);
  int out_height = C->dim32(2);
  int in_width = A.dim32(3);
  int in_height = A.dim32(2);

  CopyKernel<T><<<
      CAFFE_GET_BLOCKS(C->size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      C->dim32(0),
      C->dim32(1),
      in_height,
      in_width,
      out_height,
      out_width,
      A.template data<T>(),
      C->template mutable_data<T>());

  return true;
}

template <>
bool SpatialNarrowAsGradientOp<CUDAContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float_t, int32_t>>::call(this, Input(0));
}

template <>
template <typename T>
bool SpatialNarrowAsGradientOp<CUDAContext>::DoRunWithType() {
  auto& A = Input(0);
  auto& B = Input(1);
  auto& dC = Input(2); // Gradient of net w.r.t. output of forward op
  auto* dA = Output(0); // Gradient of net w.r.t. input to forward op

  dA->ResizeLike(A);
  math::Set<T, CUDAContext>(
      dA->size(), 0.f, dA->template mutable_data<T>(), &context_);
  int out_width = dA->dim32(3);
  int out_height = dA->dim32(2);
  int in_width = dC.dim32(3);
  int in_height = dC.dim32(2);

  CopyGradientKernel<T><<<
      CAFFE_GET_BLOCKS(dC.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      dA->dim32(0),
      dA->dim32(1),
      in_height,
      in_width,
      out_height,
      out_width,
      dC.template data<T>(),
      dA->template mutable_data<T>());

  return true;
}

REGISTER_CUDA_OPERATOR(SpatialNarrowAs, SpatialNarrowAsOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SpatialNarrowAsGradient,
    SpatialNarrowAsGradientOp<CUDAContext>);
} // namespace caffe2
