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
  internal::adagrad_update_base_inlined(N, w, g, h, nw, nh, decay, epsilon, lr);
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
  internal::adagrad_update_base_inlined(N, w, g, h, nw, nh, 1.0f, epsilon, lr);
}

// version without prefetching
decltype(adagrad_update__base) adagrad_update__avx_f16c;
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

decltype(adagrad_update_prefetch__base) adagrad_update_prefetch__avx_f16c;
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
decltype(
    adagrad_fp16_update_prefetch__base) adagrad_fp16_update_prefetch__avx_f16c;
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

} // namespace caffe2
