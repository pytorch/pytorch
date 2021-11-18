#include <cub/block/block_reduce.cuh>
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/sgd/adam_op.h"
#include "caffe2/utils/cub_namespace.cuh"

namespace caffe2 {

__global__ void AdamUpdate(
    int N,
    const float* g,
    const float* m,
    const float* v,
    float* ng,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    ng[i] = lr[0] * correction * mi / (sqrtf(vi) + eps_hat);
  }
}

template <>
void adam_update<CUDAContext>(
    int N,
    const float* g,
    const float* m,
    const float* v,
    float* ng,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr,
    CUDAContext* context) {
  AdamUpdate<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      N, g, m, v, ng, nm, nv, beta1, beta2, eps_hat, correction, lr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void AdamCompute(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    float ng = lr[0] * correction * mi / (sqrtf(vi) + eps_hat);
    nw[i] = w[i] + ng;
  }
}

template <>
void adam_compute<CUDAContext>(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr,
    CUDAContext* context) {
  AdamCompute<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      N, w, g, m, v, nw, nm, nv, beta1, beta2, eps_hat, correction, lr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void AdamComputeOutputGrad(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float* ng,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    float ngi = ng[i] = correction * mi / (sqrtf(vi) + eps_hat);
    nw[i] = w[i] + lr[0] * ngi;
  }
}

template <>
void adam_compute_output_grad<CUDAContext>(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float* ng,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr,
    CUDAContext* context) {
  AdamComputeOutputGrad<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      N, w, g, m, v, nw, nm, nv, ng, beta1, beta2, eps_hat, correction, lr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename SIndex>
__global__ void SparseAdamKernel(
    const size_t N,
    const size_t grad_slice_sz,
    const float beta1,
    const float beta2,
    const float epsilon,
    float* param,
    float* mom1,
    float* mom2,
    const SIndex* indices,
    const float* grad,
    const float correction,
    const float* lr,
    const float iter) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const size_t gradIdx = i;
    const SIndex index = indices[i / grad_slice_sz];
    const size_t paramIdx = index * grad_slice_sz + (i % grad_slice_sz);

    float m1n = mom1[paramIdx] =
        mom1[paramIdx] * beta1 + grad[gradIdx] * (1.0f - beta1);
    float m2n = mom2[paramIdx] =
        mom2[paramIdx] * beta2 + grad[gradIdx] * grad[gradIdx] * (1.0f - beta2);
    param[paramIdx] += lr[0] * correction * m1n / (sqrt(m2n) + epsilon);
  }
}

template <typename SIndex>
__global__ void SparseAdamOutputGradKernel(
    const size_t N,
    const size_t grad_slice_sz,
    const float beta1,
    const float beta2,
    const float epsilon,
    float* param,
    float* mom1,
    float* mom2,
    float* output_grad,
    const SIndex* indices,
    const float* grad,
    const float correction,
    const float* lr,
    const float iter) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    const size_t gradIdx = i;
    const SIndex index = indices[i / grad_slice_sz];
    const size_t paramIdx = index * grad_slice_sz + (i % grad_slice_sz);

    float m1n = mom1[paramIdx] =
        mom1[paramIdx] * beta1 + grad[gradIdx] * (1.0f - beta1);
    float m2n = mom2[paramIdx] =
        mom2[paramIdx] * beta2 + grad[gradIdx] * grad[gradIdx] * (1.0f - beta2);
    float gradOut = output_grad[gradIdx] =
        correction * m1n / (sqrt(m2n) + epsilon);
    param[paramIdx] += lr[0] * gradOut;
  }
}

template <typename SIndex>
__global__ void RowWiseSparseAdamKernel(
    const int M,
    const int N,
    const float beta1,
    const float beta2,
    const float epsilon,
    float* param,
    float* mom1,
    float* mom2,
    const SIndex* indices,
    const float* grad,
    const float correction,
    const float* lr) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ BlockReduce::TempStorage temp_storage;
  int valid = min(N, CAFFE_CUDA_NUM_THREADS);
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

    float reduce_sum_squares =
        BlockReduce(temp_storage).Sum(sum_squares, valid);
    if (threadIdx.x == 0) {
      row_sum_squares_avg = reduce_sum_squares / (float)N;
      mom2[index] = mom2[index] * beta2 + row_sum_squares_avg * (1.0f - beta2);
    }

    __syncthreads();
    // update param
    float step = correction / (std::sqrt(mom2[index]) + epsilon);
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      mom1[index * N + j] =
          mom1[index * N + j] * beta1 + grad[i * N + j] * (1.0f - beta1);
      param[index * N + j] += lr[0] * mom1[index * N + j] * step;
    }
  }
}

template <typename SIndex>
__global__ void RowWiseSparseAdamOutputGradKernel(
    const int M,
    const int N,
    const float beta1,
    const float beta2,
    const float epsilon,
    float* param,
    float* mom1,
    float* mom2,
    float* output_grad,
    const SIndex* indices,
    const float* grad,
    const float correction,
    const float* lr) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ BlockReduce::TempStorage temp_storage;
  int valid = min(N, CAFFE_CUDA_NUM_THREADS);
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

    float reduce_sum_squares =
        BlockReduce(temp_storage).Sum(sum_squares, valid);
    if (threadIdx.x == 0) {
      row_sum_squares_avg = reduce_sum_squares / (float)N;
      mom2[index] = mom2[index] * beta2 + row_sum_squares_avg * (1.0f - beta2);
    }

    __syncthreads();
    // update param
    float step = correction / (std::sqrt(mom2[index]) + epsilon);
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      mom1[index * N + j] =
          mom1[index * N + j] * beta1 + grad[i * N + j] * (1.0f - beta1);
      output_grad[i * N + j] = mom1[index * N + j] * step;
      param[index * N + j] += lr[0] * output_grad[i * N + j];
    }
  }
}

template <>
template <typename SIndex>
bool SparseAdamOp<float, CUDAContext>::DoRunWithType() {
  Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
  Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));
  Output(OUTPUT_MOMENT_2)->ResizeLike(Input(MOMENT_2));

  auto N = Input(GRAD).size();
  auto grad_slice_sz = Input(GRAD).size_from_dim(Input(INDICES).ndim());
  const auto iter =
      OperatorBase::Input<Tensor>(ITER, CPU).template data<int64_t>()[0];
  const float correction = sqrtf(1.0f - std::pow(beta2_, iter + 1)) /
      (1.0f - std::pow(beta1_, iter + 1));

  if (OutputSize() == 3) {
    SparseAdamKernel<SIndex>
        <<<CAFFE_GET_BLOCKS(N),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N,
            grad_slice_sz,
            beta1_,
            beta2_,
            epsilon_,
            Output(OUTPUT_PARAM)->template mutable_data<float>(),
            Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
            Output(OUTPUT_MOMENT_2)->template mutable_data<float>(),
            Input(INDICES).template data<SIndex>(),
            Input(GRAD).template data<float>(),
            correction,
            Input(LR).template data<float>(),
            iter);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
    SparseAdamOutputGradKernel<SIndex>
        <<<CAFFE_GET_BLOCKS(N),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N,
            grad_slice_sz,
            beta1_,
            beta2_,
            epsilon_,
            Output(OUTPUT_PARAM)->template mutable_data<float>(),
            Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
            Output(OUTPUT_MOMENT_2)->template mutable_data<float>(),
            Output(OUTPUT_GRAD)->template mutable_data<float>(),
            Input(INDICES).template data<SIndex>(),
            Input(GRAD).template data<float>(),
            correction,
            Input(LR).template data<float>(),
            iter);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return true;
}

template <>
template <typename SIndex>
bool RowWiseSparseAdamOp<float, CUDAContext>::DoRunWithType() {
  Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
  Output(OUTPUT_MOMENT_1)->ResizeLike(Input(MOMENT_1));
  Output(OUTPUT_MOMENT_2)->ResizeLike(Input(MOMENT_2));

  auto N = Input(GRAD).size();
  if (N == 0) {
    // empty grad, nothing to do here, not even launching the kernel
    return true;
  }
  const auto iter =
      OperatorBase::Input<Tensor>(ITER, CPU).template data<int64_t>()[0];
  const float correction = sqrtf(1.0f - std::pow(beta2_, iter + 1)) /
      (1.0f - std::pow(beta1_, iter + 1));

  // size of the 1st dimension of the input gradient
  auto GRAD_M = Input(GRAD).dim32(0);
  auto GRAD_N = N / GRAD_M;

  if (OutputSize() == 3) {
    RowWiseSparseAdamKernel<SIndex>
        <<<std::min(GRAD_M, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            GRAD_M,
            GRAD_N,
            beta1_,
            beta2_,
            epsilon_,
            Output(OUTPUT_PARAM)->template mutable_data<float>(),
            Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
            Output(OUTPUT_MOMENT_2)->template mutable_data<float>(),
            Input(INDICES).template data<SIndex>(),
            Input(GRAD).template data<float>(),
            correction,
            Input(LR).template data<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    Output(OUTPUT_GRAD)->ResizeLike(Input(GRAD));
    RowWiseSparseAdamOutputGradKernel<SIndex>
        <<<std::min(GRAD_M, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            GRAD_M,
            GRAD_N,
            beta1_,
            beta2_,
            epsilon_,
            Output(OUTPUT_PARAM)->template mutable_data<float>(),
            Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
            Output(OUTPUT_MOMENT_2)->template mutable_data<float>(),
            Output(OUTPUT_GRAD)->template mutable_data<float>(),
            Input(INDICES).template data<SIndex>(),
            Input(GRAD).template data<float>(),
            correction,
            Input(LR).template data<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return true;
}

REGISTER_CUDA_OPERATOR(Adam, AdamOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SparseAdam, SparseAdamOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    RowWiseSparseAdam,
    RowWiseSparseAdamOp<float, CUDAContext>);
} // namespace caffe2
