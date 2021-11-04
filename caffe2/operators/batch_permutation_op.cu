#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/batch_permutation_op.h"

namespace caffe2 {

namespace {
template <bool forward>
__global__ void BatchPermutationKernel(
    int N,
    int K,
    const float* src,
    const int* indices,
    float* dst) {
  if (forward) {
    CUDA_1D_KERNEL_LOOP(index, N * K) {
      int k = index % K;
      int n = index / K;
      int idx = indices[n];
      CUDA_KERNEL_ASSERT(idx >= 0);
      CUDA_KERNEL_ASSERT(idx < N);
      dst[index] = src[idx * K + k];
    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, N * K) {
      int k = index % K;
      int n = index / K;

      // NOTE: an alternative implementation if we want to align the index with
      // the output tensor (rather than the input tensor).
      // int idx = -1;
      // for (size_t i = 0; i < N; ++i) {
      //   if (indices[i] == n) {
      //     idx = i;
      //   }
      // }
      // CUDA_KERNEL_ASSERT(idx >= 0);
      // CUDA_KERNEL_ASSERT(idx < N);
      // dst[index] = src[idx * K + k];

      int idx = indices[n];
      CUDA_KERNEL_ASSERT(idx >= 0);
      CUDA_KERNEL_ASSERT(idx < N);
      dst[idx * K + k] = src[index];
    }
  }
}
} // namespace

template <>
bool BatchPermutationOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& indices = Input(1);

  CAFFE_ENFORCE(indices.dim() == 1, "indices must be 1-d");
  CAFFE_ENFORCE(
      X.dim32(0) == indices.dim32(0),
      "X.dim32(0) must be equal to indices.dim32(0)",
      "(",
      X.dim32(0),
      " vs. ",
      indices.dim32(0),
      ")");

  auto* Y = Output(0, X.sizes(), at::dtype<float>());

  if (X.dim32(0) > 0) {
    BatchPermutationKernel<true>
        <<<CAFFE_GET_BLOCKS(X.numel()),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            X.dim32(0),
            X.numel() / X.dim32(0),
            X.data<float>(),
            indices.data<int>(),
            Y->mutable_data<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return true;
}

template <>
bool BatchPermutationGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& indices = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0, dY.sizes(), at::dtype<float>());

  if (dY.dim32(0) > 0) {
    BatchPermutationKernel<false>
        <<<CAFFE_GET_BLOCKS(dY.numel()),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            dY.dim32(0),
            dY.numel() / dY.dim32(0),
            dY.data<float>(),
            indices.data<int>(),
            dX->mutable_data<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return true;
}

REGISTER_CUDA_OPERATOR(
    BatchPermutation,
    BatchPermutationOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    BatchPermutationGradient,
    BatchPermutationGradientOp<float, CUDAContext>);
} // namespace caffe2

using BatchPermutationOpFloatCUDA =
    caffe2::BatchPermutationOp<float, caffe2::CUDAContext>;

C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(BatchPermutation, BatchPermutationOpFloatCUDA);
