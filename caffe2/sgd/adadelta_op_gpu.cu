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
#include <cub/cub.cuh>
#include "adadelta_op.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/mixed_utils.h"

namespace caffe2 {

__global__ void AdadeltaUpdate(
    int N,
    const float* w,
    const float* g,
    const float* h,
    const float* p,
    float* nw,
    float* nh,
    float* np,
    float epsilon,
    float decay) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = nh[i] = decay * h[i] + (1.0f - decay) * gi * gi;
    float pi = gi * (std::sqrt(p[i]) + epsilon) / (std::sqrt(hi) + epsilon);
    np[i] = decay * p[i] + (1.0f - decay) * pi * pi;
    nw[i] = w[i] - pi;
  }
}

template <>
void adadelta_update<CUDAContext>(
    int N,
    const float* w,
    const float* g,
    const float* h,
    const float* p,
    float* nw,
    float* nh,
    float* np,
    float epsilon,
    float decay,
    CUDAContext* context) {
  AdadeltaUpdate<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(N, w, g, h, p, nw, nh, np, epsilon, decay);
}

template <typename SIndex, typename THalf>
__global__ void SparseAdadeltaKernel(
    const size_t N,
    const size_t grad_slice_sz,
    const float epsilon,
    const float decay,
    THalf* param,
    THalf* param_mom,
    THalf* param_mom_delta,
    const SIndex* indices,
    const float* grad) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const size_t gradIdx = i;
    const SIndex index = indices[i / grad_slice_sz];
    const size_t paramIdx = index * grad_slice_sz + (i % grad_slice_sz);
    float mom_new = (1.0f - decay) * grad[gradIdx] * grad[gradIdx] +
        mixed_mult(decay, param_mom[paramIdx]);
    mixed_store(&mom_new, &(param_mom[paramIdx]));
    float grad_new = grad[gradIdx] *
        (sqrt(mixed_add(0.0f, param_mom_delta[paramIdx])) + epsilon) /
        (sqrt(mom_new) + epsilon);
    float param_new = mixed_add(-1.0f * grad_new, param[paramIdx]);
    mixed_store(&param_new, &(param[paramIdx]));
    float mom2_new = (1.0f - decay) * grad_new * grad_new +
        mixed_mult(decay, param_mom_delta[paramIdx]);
    mixed_store(&mom2_new, &(param_mom_delta[paramIdx]));
  }
}

/**
 * Calculate RowwiseSparseAdadelta
 * M: gradients.dims[0]
 * N: gradients.size_from_dim(1)
 * grad: pointer to the gradients
 * param: pointer to weights
 * param_mom: pointer to the momentum
 * indices: keys
 */
template <typename SIndex>
__global__ void RowWiseSparseAdadeltaKernel(
    const int M,
    const int N,
    const float epsilon,
    const float decay,
    float* param,
    float* param_mom,
    float* param_mom_delta,
    const SIndex* indices,
    const float* grad) {
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
      param_mom[index] =
          decay * param_mom[index] + (1.0f - decay) * row_sum_squares_avg;
    }
    __syncthreads();
    // update param
    float step = (std::sqrt(param_mom_delta[index]) + epsilon) /
        (std::sqrt(param_mom[index]) + epsilon);
    float delta_sum_sq = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      float delta = grad[i * N + j] * step;
      param[index * N + j] = param[index * N + j] - delta;
      delta_sum_sq += delta * delta;
    }
    param_mom_delta[index] = decay * param_mom_delta[index] +
        (1.0f - decay) * delta_sum_sq / (float)N;
  }
}

template <typename T, class Context>
class CUDASparseAdadeltaOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseAdadeltaOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)),
        decay_(OperatorBase::GetSingleArgument<float>("decay", 1.0f)) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_GRAD).size());
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_DELTA).size());
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    auto n = Input(INDICES).size();
    if (n == 0) {
      return true;
    }
    return DispatchHelper<TensorTypes2<float, float16>, IndexType>::call(
        this, Input(PARAM));
  }

  template <typename IndexType, typename THalf>
  bool DoRunWithType2() {
    const auto* indices = Input(INDICES).template data<IndexType>();
    const auto* gradIn = Input(GRAD).template data<T>();
    const auto* paramIn = Input(PARAM).template data<THalf>();
    const auto* momentIn = Input(MOMENT_GRAD).template data<THalf>();
    const auto* moment2In = Input(MOMENT_DELTA).template data<THalf>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<THalf>();
    auto* momentOut =
        Output(OUTPUT_MOMENT_GRAD)->template mutable_data<THalf>();
    auto* moment2Out =
        Output(OUTPUT_MOMENT_DELTA)->template mutable_data<THalf>();

    auto N = Input(GRAD).size();
    auto grad_slice_sz = Input(GRAD).size_from_dim(Input(INDICES).ndim());
    if (N == 0) {
      // empty grad, nothing to do here, not even launching the kernel
      return true;
    }
    SparseAdadeltaKernel<IndexType, THalf>
        <<<CAFFE_GET_BLOCKS(N),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N,
            grad_slice_sz,
            epsilon_,
            decay_,
            Output(OUTPUT_PARAM)->template mutable_data<THalf>(),
            Output(OUTPUT_MOMENT_GRAD)->template mutable_data<THalf>(),
            Output(OUTPUT_MOMENT_DELTA)->template mutable_data<THalf>(),
            Input(INDICES).template data<IndexType>(),
            Input(GRAD).template data<float>());
    return true;
  }

 protected:
  T epsilon_;
  T decay_;
  INPUT_TAGS(PARAM, MOMENT_GRAD, MOMENT_DELTA, INDICES, GRAD);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_GRAD, OUTPUT_MOMENT_DELTA);
};

template <>
template <typename SIndex>
bool RowWiseSparseAdadeltaOp<float, CUDAContext>::DoRunWithType() {
  auto N = Input(GRAD).size();
  if (N == 0) {
    // empty grad, nothing to do here, not even launching the kernel
    return true;
  }
  // size of the 1st dimension of the input gradient
  auto GRAD_M = Input(GRAD).dim32(0);
  auto GRAD_N = N / GRAD_M;

  // each thread block will handle multiple rows of the input and output
  RowWiseSparseAdadeltaKernel<<<
      min(GRAD_M, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      GRAD_M,
      GRAD_N,
      epsilon_,
      decay_,
      Output(OUTPUT_PARAM)->template mutable_data<float>(),
      Output(OUTPUT_MOMENT_GRAD)->template mutable_data<float>(),
      Output(OUTPUT_MOMENT_DELTA)->template mutable_data<float>(),
      Input(INDICES).template data<SIndex>(),
      Input(GRAD).template data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(Adadelta, AdadeltaOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SparseAdadelta,
    CUDASparseAdadeltaOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    RowWiseSparseAdadelta,
    RowWiseSparseAdadeltaOp<float, CUDAContext>);
} // namespace caffe2
