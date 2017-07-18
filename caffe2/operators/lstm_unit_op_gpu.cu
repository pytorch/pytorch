#include <algorithm>
#include <cmath>
#include <vector>
#include "caffe2/core/context_gpu.h"
#include "lstm_unit_op.h"

namespace caffe2 {

namespace detail {

template <typename Dtype>
__device__ Dtype cuda_sigmoid(const Dtype x) {
  return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename T>
__global__ void LSTMUnitKernel(
    const int nthreads,
    const int dim,
    const int t,
    const T* H_prev,
    const T* C_prev,
    const T* X,
    const int32_t* seqLengths,
    bool drop_states,
    T* C,
    T* H,
    const T forget_bias) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int d = index % dim;
    const bool valid = t < seqLengths[n];
    if (!valid) {
      H[index] = H_prev[index] * !drop_states;
      C[index] = C_prev[index] * !drop_states;
    } else {
      const T* X_offset = X + 4 * dim * n;
      const T i = cuda_sigmoid(X_offset[d]);
      const T f = cuda_sigmoid(X_offset[1 * dim + d] + forget_bias);
      const T o = cuda_sigmoid(X_offset[2 * dim + d]);
      const T g = tanh(X_offset[3 * dim + d]);
      const T c_prev = C_prev[index];
      const T c = f * c_prev + i * g;
      C[index] = c;
      const T tanh_c = tanh(c);
      H[index] = o * tanh_c;
    }
  }
}

template <typename T>
__global__ void LSTMUnitGradientKernel(
    const int nthreads,
    const int dim,
    const int t,
    const T* C_prev,
    const T* X,
    const T* C,
    const T* H,
    const int32_t* seqLengths,
    const T* C_diff,
    const T* H_diff,
    bool drop_states,
    T* H_prev_diff,
    T* C_prev_diff,
    T* X_diff,
    const T forget_bias) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const bool valid = t < seqLengths[n];
    const int d = index % dim;
    const T* X_offset = X + 4 * dim * n;
    T* c_prev_diff = C_prev_diff + index;
    T* h_prev_diff = H_prev_diff + index;
    T* X_diff_offset = X_diff + 4 * dim * n;
    T* i_diff = X_diff_offset + d;
    T* f_diff = X_diff_offset + 1 * dim + d;
    T* o_diff = X_diff_offset + 2 * dim + d;
    T* g_diff = X_diff_offset + 3 * dim + d;
    if (!valid) {
      *h_prev_diff = H_diff[index] * !drop_states;
      *c_prev_diff = C_diff[index] * !drop_states;
      *i_diff = 0;
      *f_diff = 0;
      *o_diff = 0;
      *g_diff = 0;
    } else {
      const T i = cuda_sigmoid(X_offset[d]);
      const T f = cuda_sigmoid(X_offset[1 * dim + d] + forget_bias);
      const T o = cuda_sigmoid(X_offset[2 * dim + d]);
      const T g = tanh(X_offset[3 * dim + d]);
      const T c_prev = C_prev[index];
      const T c = C[index];
      const T tanh_c = tanh(c);
      const T c_term_diff =
          C_diff[index] + H_diff[index] * o * (1 - tanh_c * tanh_c);
      *c_prev_diff = c_term_diff * f;
      *h_prev_diff = 0;
      *i_diff = c_term_diff * g * i * (1 - i);
      *f_diff = c_term_diff * c_prev * f * (1 - f);
      *o_diff = H_diff[index] * tanh_c * o * (1 - o);
      *g_diff = c_term_diff * i * (1 - g * g);
    }
  }
}

template <>
void LSTMUnit<float, CUDAContext>(
    int N,
    int D,
    int t,
    const float* H_prev,
    const float* C_prev,
    const float* X,
    const int32_t* seqLengths,
    bool drop_states,
    float* C,
    float* H,
    const float forget_bias,
    CUDAContext* context) {
  LSTMUnitKernel<float><<<
      CAFFE_GET_BLOCKS(N * D),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      N * D,
      D,
      t,
      H_prev,
      C_prev,
      X,
      seqLengths,
      drop_states,
      C,
      H,
      forget_bias);
}

template <>
void LSTMUnitGradient<float, CUDAContext>(
    int N,
    int D,
    int t,
    const float* C_prev,
    const float* X,
    const int32_t* seqLengths,
    const float* C,
    const float* H,
    const float* C_diff,
    const float* H_diff,
    bool drop_states,
    float* H_prev_diff,
    float* C_prev_diff,
    float* X_diff,
    const float forget_bias,
    CUDAContext* context) {
  LSTMUnitGradientKernel<float><<<
      CAFFE_GET_BLOCKS(N * D),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      N * D,
      D,
      t,
      C_prev,
      X,
      C,
      H,
      seqLengths,
      C_diff,
      H_diff,
      drop_states,
      H_prev_diff,
      C_prev_diff,
      X_diff,
      forget_bias);
}
}

namespace {
REGISTER_CUDA_OPERATOR(LSTMUnit, LSTMUnitOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    LSTMUnitGradient,
    LSTMUnitGradientOp<float, CUDAContext>);
}
}
