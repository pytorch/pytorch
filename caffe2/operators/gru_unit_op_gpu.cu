#include <algorithm>
#include <cmath>
#include <vector>
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/gru_unit_op.h"

namespace caffe2 {

namespace detail {

template <typename Dtype>
__device__ Dtype cuda_sigmoid(const Dtype x) {
  return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename T>
__global__ void GRUUnitKernel(
    const int ND,
    const int dim,
    const int t,
    const T* H_prev,
    const T* X,
    const int32_t* seqLengths,
    bool drop_states,
    T* H) {
  // index is virtual thread ID in range [0, ND)
  CUDA_1D_KERNEL_LOOP(index, ND) {
    const int n = index / dim;
    const int d = index % dim;
    const bool valid = seqLengths == nullptr || t < seqLengths[n];
    if (!valid) {
      H[index] = H_prev[index] * !drop_states;
    } else {
      const T* X_offset = X + 3 * dim * n;
      const T update = X_offset[1 * dim + d];
      const T output = X_offset[2 * dim + d];
      T sigmoid_update = cuda_sigmoid(update);
      H[index] = H_prev[index] * sigmoid_update +
          tanh(output) * (1.0f - sigmoid_update);
    }
  }
}

template <typename T>
__global__ void GRUUnitGradientKernel(
    const int ND,
    const int dim,
    const int t,
    const T* H_prev,
    const T* X,
    const int32_t* seqLengths,
    const T* H,
    const T* H_diff,
    bool drop_states,
    T* H_prev_diff,
    T* X_diff) {
  CUDA_1D_KERNEL_LOOP(index, ND) {
    const int n = index / dim;
    const bool valid = seqLengths == nullptr || t < seqLengths[n];
    const int d = index % dim;
    const T* X_offset = X + 3 * dim * n;
    T* h_prev_diff = H_prev_diff + index;
    T* X_diff_offset = X_diff + 3 * dim * n;
    T* reset_diff = X_diff_offset + 0 * dim + d;
    T* update_diff = X_diff_offset + 1 * dim + d;
    T* output_diff = X_diff_offset + 2 * dim + d;

    if (!valid) {
      *h_prev_diff = H_diff[index] * !drop_states;
      *reset_diff = 0;
      *update_diff = 0;
      *output_diff = 0;
    } else {
      const T u = cuda_sigmoid(X_offset[1 * dim + d]);
      const T o = tanh(X_offset[2 * dim + d]);

      *h_prev_diff = H_diff[index] * u;
      *reset_diff = 0; // 0 contribution to gradient from this operation
      *update_diff =
          (H_diff[index] * H_prev[index] - H_diff[index] * o) * u * (1.0f - u);
      *output_diff = H_diff[index] * (1.0f - u) * (1.0f - o * o);
    }
  }
}

template <>
void GRUUnit<float, CUDAContext>(
    int N,
    int D,
    int t,
    const float* H_prev,
    const float* X,
    const int32_t* seqLengths,
    bool drop_states,
    float* H,
    CUDAContext* context) {
  GRUUnitKernel<float>
      <<<CAFFE_GET_BLOCKS(N * D),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          N * D, D, t, H_prev, X, seqLengths, drop_states, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <>
void GRUUnitGradient<float, CUDAContext>(
    int N,
    int D,
    int t,
    const float* H_prev,
    const float* X,
    const int32_t* seqLengths,
    const float* H,
    const float* H_diff,
    bool drop_states,
    float* H_prev_diff,
    float* X_diff,
    CUDAContext* context) {
  GRUUnitGradientKernel<float>
      <<<CAFFE_GET_BLOCKS(N * D),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          N * D,
          D,
          t,
          H_prev,
          X,
          seqLengths,
          H,
          H_diff,
          drop_states,
          H_prev_diff,
          X_diff);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

REGISTER_CUDA_OPERATOR(GRUUnit, GRUUnitOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(GRUUnitGradient, GRUUnitGradientOp<float, CUDAContext>);
}
