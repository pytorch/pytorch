#include "caffe2/perfkernels/adagrad.h"

#include <cmath>

#include "caffe2/perfkernels/common.h"

namespace caffe2 {

void adagrad_update__base(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    float decay,
    const float lr) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float hi = nh[i] = decay * h[i] + gi * gi;
    nw[i] = w[i] + lr * gi / (std::sqrt(hi) + epsilon);
  }
}

void adagrad_update_prefetch__base(
    int N,
    const float* w,
    const float* /* w_n */, // prefetch ptr

    const float* g,

    const float* h,
    const float* /* h_n */, // prefetch ptr

    float* nw,
    float* /* nw_n */, // prefetch ptr

    float* nh,
    float* /* nh_n */, // prefetch ptr

    float epsilon,
    float lr) {
  adagrad_update__base(N, w, g, h, nw, nh, epsilon, 1.0f, lr);
}

void adagrad_fp16_update_prefetch__base(
    int N,
    const at::Half* w,
    const at::Half* /* w_n */, // prefetch ptr
    const float* g,
    const at::Half* h,
    const at::Half* /* h_n */, // prefetch ptr
    at::Half* nw,
    at::Half* /* nw_n */, // prefetch ptr
    at::Half* nh,
    at::Half* /* nh_n */, // prefetch ptr
    float epsilon,
    float lr) {
  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    float hi = h[i] + gi * gi;
    nh[i] = hi;
    nw[i] = w[i] + lr * gi / (std::sqrt(hi) + epsilon);
  }
}

void rowwise_adagrad_update__base(
    int N,
    float* w,
    float* /* w_n */, // prefetch ptr

    const float* g,

    float* h,
    float* /* h_n */, // prefetch ptr

    float epsilon,
    float lr) {
  float sum = 0.0f;
  for (auto i = 0; i < N; ++i) {
    sum += g[i] * g[i];
  }
  sum /= N;

  float hi = *h = *h + sum;
  float float_step = lr / (std::sqrt(hi) + epsilon);

  for (auto i = 0; i < N; ++i) {
    float gi = g[i];
    w[i] = w[i] + gi * float_step;
  }
}

void adagrad_update_prefetch(
    int N,
    const float* w,
    const float* w_n, // prefetch ptr

    const float* g,

    const float* h,
    const float* h_n, // prefetch ptr

    float* nw,
    float* nw_n, // prefetch ptr

    float* nh,
    float* nh_n, // prefetch ptr

    float epsilon,
    float lr) {
  AVX_F16C_DO(
      adagrad_update_prefetch,
      N,
      w,
      w_n,
      g,
      h,
      h_n,
      nw,
      nw_n,
      nh,
      nh_n,
      epsilon,
      lr);
  BASE_DO(
      adagrad_update_prefetch,
      N,
      w,
      w_n,
      g,
      h,
      h_n,
      nw,
      nw_n,
      nh,
      nh_n,
      epsilon,
      lr);
}

// Version with prefetching for embeddings and
// momentum using fp16
void adagrad_fp16_update_prefetch(
    int N,
    const at::Half* w,
    const at::Half* w_n, // prefetch ptr
    const float* g,
    const at::Half* h,
    const at::Half* h_n, // prefetch ptr
    at::Half* nw,
    at::Half* nw_n, // prefetch ptr
    at::Half* nh,
    at::Half* nh_n, // prefetch ptr
    float epsilon,
    float lr) {
  AVX_F16C_DO(
      adagrad_fp16_update_prefetch,
      N,
      w,
      w_n,
      g,
      h,
      h_n,
      nw,
      nw_n,
      nh,
      nh_n,
      epsilon,
      lr);
  BASE_DO(
      adagrad_fp16_update_prefetch,
      N,
      w,
      w_n,
      g,
      h,
      h_n,
      nw,
      nw_n,
      nh,
      nh_n,
      epsilon,
      lr);
}

void rowwise_adagrad_update(
    int N,
    float* w,
    float* w_n, // prefetch ptr

    const float* g,

    float* h,
    float* h_n, // prefetch ptr

    float epsilon,
    float lr) {
  AVX_F16C_DO(rowwise_adagrad_update, N, w, w_n, g, h, h_n, epsilon, lr);
  BASE_DO(rowwise_adagrad_update, N, w, w_n, g, h, h_n, epsilon, lr);
}

// version without prefetching
void adagrad_update(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    float decay,
    float lr) {
  AVX_F16C_DO(adagrad_update, N, w, g, h, nw, nh, epsilon, decay, lr);
  BASE_DO(adagrad_update, N, w, g, h, nw, nh, epsilon, decay, lr);
}

template <typename SIndex>
void sparse_adagrad(
    int num_rows,
    int block_size,
    int param_size,
    const float* w,
    const float* g,
    const float* h,
    const SIndex* indices,
    float* nw,
    float* nh,
    float epsilon,
    float lr,
    const std::string& param_name);

SPARSE_ADAGRAD_SPECIALIZATION(int32_t, base);

template <>
void sparse_adagrad(
    int num_rows,
    int block_size,
    int param_size,
    const float* w,
    const float* g,
    const float* h,
    const int32_t* indices,
    float* nw,
    float* nh,
    float epsilon,
    float lr,
    const std::string& param_name) {
  AVX_F16C_DO(
      sparse_adagrad_int32_t,
      num_rows,
      block_size,
      param_size,
      w,
      g,
      h,
      indices,
      nw,
      nh,
      epsilon,
      lr,
      param_name);
  BASE_DO(
      sparse_adagrad_int32_t,
      num_rows,
      block_size,
      param_size,
      w,
      g,
      h,
      indices,
      nw,
      nh,
      epsilon,
      lr,
      param_name);
}

SPARSE_ADAGRAD_SPECIALIZATION(int64_t, base);

template <>
void sparse_adagrad(
    int num_rows,
    int block_size,
    int param_size,
    const float* w,
    const float* g,
    const float* h,
    const int64_t* indices,
    float* nw,
    float* nh,
    float epsilon,
    float lr,
    const std::string& param_name) {
  AVX_F16C_DO(
      sparse_adagrad_int64_t,
      num_rows,
      block_size,
      param_size,
      w,
      g,
      h,
      indices,
      nw,
      nh,
      epsilon,
      lr,
      param_name);
  BASE_DO(
      sparse_adagrad_int64_t,
      num_rows,
      block_size,
      param_size,
      w,
      g,
      h,
      indices,
      nw,
      nh,
      epsilon,
      lr,
      param_name);
}

} // namespace caffe2
