#pragma once

#include "caffe2/core/types.h"

namespace caffe2 {

// version with prefetching
// TODO(msmelyan)
// Crux of the computation is computing a  / (sqrt(b) + epsilon),
// where a and b are vectors and epislon is very small (eg., 10^-5) and does not
// change. Today it's computed using two vector sqrt and vector divide simd
// instructions. It is slow. We can take advantage of existing fast vector
// VRSQRTPS instruction that computes approximate reciprocals of square roots
// of the vector. It is 6x faster than vsrt and vdiv combinations. Since the
// addition of epislon is just done to avoid division by zero, we approximate a
// / (sqrt(b) + epsilon) by a / (sqrt(b + sqrt(epsilon)) If we do that, we can
// use VRSQRTPS instead now. VRSQRTPS is not very accurate. Specifically, for
// the test on random numbers between 0.1 and 1 the absolute error was about
// 10^-3 compared to using slower but more accurate combination of vsqrt and
// vdiv. Extend Marat's function with more NR iterations to get more accuracy
// for training
// TODO(msmelyan)
// explore streaming stores, but need to have inuque indices (deduplication)
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
    float lr);

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
    float lr);

void rowwise_adagrad_update(
    int N,
    float* w,
    float* w_n, // prefetch ptr

    const float* g,

    float* h,
    float* h_n, // prefetch ptr

    float epsilon,
    float lr);

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
    float lr);

template <typename SIndex>
void sparse_adagrad(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    int param_size, // total number of parameters
    const float* w, // input parameters
    const float* g, // input gradients
    const float* h, // input momentums
    const SIndex* indices, // indices of each row
    float* nw, // output parameters
    float* nh, // output momentums
    float epsilon,
    float lr,
    const std::string& param_name); // name of parameters (for error reporting)

#define SPARSE_ADAGRAD_SPECIALIZATION(SIndex, ISA)                       \
  void sparse_adagrad_##SIndex##__##ISA(                                 \
      int num_rows,                                                      \
      int block_size,                                                    \
      int param_size,                                                    \
      const float* w,                                                    \
      const float* g,                                                    \
      const float* h,                                                    \
      const SIndex* indices,                                             \
      float* nw,                                                         \
      float* nh,                                                         \
      float epsilon,                                                     \
      float lr,                                                          \
      const std::string& param_name) {                                   \
    for (int i = 0; i < num_rows; ++i) {                                 \
      auto idx = indices[i];                                             \
      auto offsetI = i * block_size;                                     \
      auto offsetIdx = idx * block_size;                                 \
                                                                         \
      CAFFE_ENFORCE_GE(                                                  \
          param_size,                                                    \
          block_size + offsetIdx,                                        \
          param_name,                                                    \
          ", out of bound,  idx:",                                       \
          idx,                                                           \
          " for input i:",                                               \
          i,                                                             \
          " and block size:",                                            \
          block_size,                                                    \
          " max size:",                                                  \
          param_size);                                                   \
                                                                         \
      if (block_size == 1) {                                             \
        float gi = g[i];                                                 \
        float hi = nh[idx] = h[idx] + gi * gi;                           \
        nw[idx] = w[idx] + lr * gi / (std::sqrt(hi) + epsilon);          \
      } else {                                                           \
        const int prefdist_T0 = 16;                                      \
        int i_pref = (i < num_rows - prefdist_T0) ? i + prefdist_T0 : i; \
        uint64_t idx_pref = indices[i_pref];                             \
                                                                         \
        adagrad_update_prefetch__##ISA(                                  \
            block_size,                                                  \
            w + offsetIdx,                                               \
            &w[idx_pref * block_size],                                   \
            g + offsetI,                                                 \
            h + offsetIdx,                                               \
            &h[idx_pref * block_size],                                   \
            nw + offsetIdx,                                              \
            &nw[idx_pref * block_size],                                  \
            nh + offsetIdx,                                              \
            &nh[idx_pref * block_size],                                  \
            epsilon,                                                     \
            lr);                                                         \
      }                                                                  \
    }                                                                    \
  };

} // namespace caffe2
