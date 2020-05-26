#pragma once

#if defined(__AVX__) && !defined(__NVCC__) && \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__))
#define CAFFE2_PERFKERNELS_ADAGRAD_H_USE_INTRINSIC
#include <immintrin.h>
#endif
#include <c10/util/Half.h>

namespace caffe2 {

namespace internal {

// The following functions inside internal namespace are inlined because they
// are performance critical.

template <typename T>
static inline void adagrad_update_base_inlined(
    int N,
    const T* w,
    const float* g,
    const T* h,
    T* nw,
    T* nh,
    float decay,
    float epsilon,
    float lr,
    float weight_decay = 0.f) {
  for (auto i = 0; i < N; ++i) {
    float gi = std::fma(weight_decay, w[i], g[i]);
    float hi = decay * h[i] + gi * gi;
    nh[i] = hi;
    nw[i] = w[i] + lr * gi / (std::sqrt(hi) + epsilon);
  }
}

// version with prefetching
// TODO(msmelyan)
// Crux of the computation is computing a  / (sqrt(b) + epsilon),
// where a and b are vectors and epsilon is very small (eg., 10^-5) and does not
// change. Today it's computed using two vector sqrt and vector divide simd
// instructions. It is slow. We can take advantage of existing fast vector
// VRSQRTPS instruction that computes approximate reciprocals of square roots
// of the vector. It is 6x faster than vsrt and vdiv combinations. Since the
// addition of epsilon is just done to avoid division by zero, we approximate a
// / (sqrt(b) + epsilon) by a / (sqrt(b + sqrt(epsilon)) If we do that, we can
// use VRSQRTPS instead now. VRSQRTPS is not very accurate. Specifically, for
// the test on random numbers between 0.1 and 1 the absolute error was about
// 10^-3 compared to using slower but more accurate combination of vsqrt and
// vdiv. Extend Marat's function with more NR iterations to get more accuracy
// for training
// TODO(msmelyan)
// explore streaming stores, but need to have unique indices (deduplication)
inline void adagrad_update_prefetch_inlined(
    int N,
    const float* w,
#ifdef CAFFE2_PERFKERNELS_ADAGRAD_H_USE_INTRINSIC
    const float* w_n, // prefetch ptr
#else
    const float* /* unused */,
#endif

    const float* g,

    const float* h,
#ifdef CAFFE2_PERFKERNELS_ADAGRAD_H_USE_INTRINSIC
    const float* h_n, // prefetch ptr
#else
    const float* /* unused */,
#endif

    float* nw,
#ifdef CAFFE2_PERFKERNELS_ADAGRAD_H_USE_INTRINSIC
    float* nw_n, // prefetch ptr
#else
    float* /* unused */,
#endif

    float* nh,
#ifdef CAFFE2_PERFKERNELS_ADAGRAD_H_USE_INTRINSIC
    float* nh_n, // prefetch ptr
#else
    float* /* unused */,
#endif

    float epsilon,
    float lr,
    float weight_decay = 0.f) {
  auto i = 0;

#ifdef CAFFE2_PERFKERNELS_ADAGRAD_H_USE_INTRINSIC
  constexpr int kSize = 8;
  for (; i + kSize <= N; i += kSize) {
    _mm_prefetch(reinterpret_cast<const char*>(&w_n[i]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&h_n[i]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&nw_n[i]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&nh_n[i]), _MM_HINT_T0);

    __m256 gi = _mm256_loadu_ps(g + i);
    __m256 hi = _mm256_loadu_ps(h + i);
    __m256 wi = _mm256_loadu_ps(w + i);
#ifdef __AVX2__
    gi = _mm256_fmadd_ps(_mm256_set1_ps(weight_decay), wi, gi);
#else
    gi = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(weight_decay), wi), gi);
#endif

    __m256 nhi = _mm256_add_ps(hi, _mm256_mul_ps(gi, gi));
    _mm256_storeu_ps(nh + i, nhi);
    __m256 vtmp = _mm256_div_ps(
        _mm256_mul_ps(_mm256_set1_ps(lr), gi),
        _mm256_add_ps(_mm256_sqrt_ps(nhi), _mm256_set1_ps(epsilon)));
    _mm256_storeu_ps(nw + i, _mm256_add_ps(wi, vtmp));
  }
#endif

  adagrad_update_base_inlined(
      N - i,
      w + i,
      g + i,
      h + i,
      nw + i,
      nh + i,
      1.0f,
      epsilon,
      lr,
      weight_decay);
}

} // namespace internal

// version with prefetching
// TODO(msmelyan)
// Crux of the computation is computing a  / (sqrt(b) + epsilon),
// where a and b are vectors and epsilon is very small (eg., 10^-5) and does not
// change. Today it's computed using two vector sqrt and vector divide simd
// instructions. It is slow. We can take advantage of existing fast vector
// VRSQRTPS instruction that computes approximate reciprocals of square roots
// of the vector. It is 6x faster than vsrt and vdiv combinations. Since the
// addition of epsilon is just done to avoid division by zero, we approximate a
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
    float lr,
    float weight_decay = 0.f);

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
    float lr,
    float weight_decay = 0.f);

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
    float lr,
    float weight_decay = 0.f);

} // namespace caffe2

#ifdef CAFFE2_PERFKERNELS_ADAGRAD_H_USE_INTRINSIC
#undef CAFFE2_PERFKERNELS_ADAGRAD_H_USE_INTRINSIC
#endif
