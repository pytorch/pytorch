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

#include <cub/block/block_reduce.cuh>
#include "adagrad_op.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
namespace caffe2 {

__global__ void AdagradUpdate(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    float decay,
    const float* lr) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = nh[i] = decay * h[i] + gi * gi;
    nw[i] = w[i] + lr[0] * gi / (std::sqrt(hi) + epsilon);
  }
}

template <>
void adagrad_update<CUDAContext>(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    float decay,
    const float* lr,
    CUDAContext* context) {
  AdagradUpdate<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(N, w, g, h, nw, nh, epsilon, decay, lr);
}

template <typename SIndex>
__global__ void SparseAdagradKernel(
    const size_t N,
    const size_t grad_slice_sz,
    const float epsilon,
    float *param,
    float *param_mom,
    const SIndex *indices,
    const float *grad,
    const float *lr)
{
  const float LR = lr[0];
  CUDA_1D_KERNEL_LOOP(i, N)
  {
    const size_t gradIdx = i;
    const SIndex index = indices[i / grad_slice_sz];
    const size_t paramIdx = index * grad_slice_sz + (i % grad_slice_sz);

    const float mom_new = param_mom[paramIdx] + grad[gradIdx] * grad[gradIdx];
    param_mom[paramIdx] = mom_new;
    param[paramIdx] += LR * grad[gradIdx] / (sqrt(mom_new) + epsilon);
  }
}

/**
 * Calculate RowwiseSparseAdagrad
 * M: gradients.dims[0]
 * N: gradients.size_from_dim(1)
 * grad: pointer to the gradients
 * param: pointer to weights
 * param_mom: pointer to the momentum
 * indices: keys
 */
template <typename SIndex>
__global__ void RowWiseSparseAdagradKernel(
    const int M,
    const int N,
    const float epsilon,
    float* param,
    float* param_mom,
    const SIndex* indices,
    const float* grad,
    const float* lr) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ BlockReduce::TempStorage temp_storage;
  // in case gridDim is smaller than M
  for (int i = blockIdx.x; i < M; i += gridDim.x) {
    const SIndex index = indices[i];
    float sum_squares = 0.0;
    __shared__ float row_sum_squares_avg;

    // in case N is bigger than block size which is 512 by default
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      const float x_ij = grad[i * N + j];
      sum_squares += x_ij * x_ij;
    }
    float reduce_result = BlockReduce(temp_storage).Sum(sum_squares);
    if (threadIdx.x == 0) {
      row_sum_squares_avg = reduce_result / (float)N;
      param_mom[index] += row_sum_squares_avg;
    }
    __syncthreads();
    // update param
    float step = lr[0] / (std::sqrt(param_mom[index]) + epsilon);
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      param[index * N + j] = param[index * N + j] + grad[i * N + j] * step;
    }
  }
}

template<>
template<typename SIndex>
bool SparseAdagradOp<float, CUDAContext>::DoRunWithType()
{
  auto N = Input(GRAD).size();
  auto grad_slice_sz = Input(GRAD).size_from_dim(Input(INDICES).ndim());

  if (N == 0) {
    // empty grad, nothing to do here, not even launching the kernel
    return true;
  }

  SparseAdagradKernel<SIndex><<<
    CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
    context_.cuda_stream()>>>(
        N, grad_slice_sz, epsilon_,
        Output(OUTPUT_PARAM)->template mutable_data<float>(),
        Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
        Input(INDICES).template data<SIndex>(),
        Input(GRAD).template data<float>(),
        Input(LR).template data<float>());
  return true;
}

template <>
template <typename SIndex>
bool RowWiseSparseAdagradOp<float, CUDAContext>::DoRunWithType() {
  auto N = Input(GRAD).size();
  if (N == 0) {
    // empty grad, nothing to do here, not even launching the kernel
    return true;
  }
  // size of the 1st dimension of the input gradient
  auto GRAD_M = Input(GRAD).dim32(0);
  auto GRAD_N = N / GRAD_M;

  // each thread block will handle multiple rows of the input and output
  RowWiseSparseAdagradKernel<<<
      min(GRAD_M, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      GRAD_M,
      GRAD_N,
      epsilon_,
      Output(OUTPUT_PARAM)->template mutable_data<float>(),
      Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
      Input(INDICES).template data<SIndex>(),
      Input(GRAD).template data<float>(),
      Input(LR).template data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(Adagrad, AdagradOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SparseAdagrad, SparseAdagradOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    RowWiseSparseAdagrad,
    RowWiseSparseAdagradOp<float, CUDAContext>);
}
