#pragma once

#include "caffe2/perfkernels/adagrad.h"

namespace caffe2 {

void adagrad_update_prefetch__avx_f16c(
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
    float lr);

void adagrad_fp16_update_prefetch__avx_f16c(
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
    float lr);

void rowwise_adagrad_update__avx_f16c(
    int N,
    float* w,
    float* w_n, // prefetch ptr

    const float* g,

    float* h,
    float* h_n, // prefetch ptr

    float epsilon,
    float lr);

// version without prefetching
void adagrad_update__avx_f16c(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    float decay,
    float lr);

SPARSE_ADAGRAD_SPECIALIZATION(int32_t, avx_f16c);
SPARSE_ADAGRAD_SPECIALIZATION(int64_t, avx_f16c);

} // namespace caffe2
