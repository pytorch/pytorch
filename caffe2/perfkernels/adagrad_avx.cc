#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/types.h"
#include "caffe2/perfkernels/adagrad.h"
#include "caffe2/perfkernels/cvtsh_ss_bugfix.h"

#include <emmintrin.h>
#include <immintrin.h>

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
// explore streaming stores, but need to have unique indices (deduplication)
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
    float lr) {
  constexpr size_t kSize = 8;
  auto i = 0;
  for (; i + kSize <= N; i += kSize) {
    _mm_prefetch(&w_n[i], _MM_HINT_T0);
    _mm_prefetch(&h_n[i], _MM_HINT_T0);
    _mm_prefetch(&nw_n[i], _MM_HINT_T0);
    _mm_prefetch(&nh_n[i], _MM_HINT_T0);

    __m256 gi = _mm256_loadu_ps(g + i);
    __m256 hi = _mm256_loadu_ps(h + i);
    __m256 wi = _mm256_loadu_ps(w + i);

    __m256 nhi = _mm256_add_ps(hi, _mm256_mul_ps(gi, gi));
    _mm256_storeu_ps(nh + i, nhi);
    __m256 vtmp = _mm256_div_ps(
        gi, _mm256_add_ps(_mm256_sqrt_ps(nhi), _mm256_set1_ps(epsilon)));
    _mm256_storeu_ps(
        nw + i, _mm256_add_ps(wi, _mm256_mul_ps(_mm256_set1_ps(lr), vtmp)));
  }

  for (; i < N; ++i) {
    float gi = g[i];
    float hi = nh[i] = h[i] + gi * gi;
    nw[i] = w[i] + lr * gi / (std::sqrt(hi) + epsilon);
  }
}

// Compute adagrad sparse, assumes embedding and momentum are at::Half
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
    float lr) {
  constexpr size_t kSize = 8;
  auto i = 0;
  for (; i + kSize <= N; i += kSize) {
    _mm_prefetch(&w_n[i], _MM_HINT_T0);
    _mm_prefetch(&h_n[i], _MM_HINT_T0);
    _mm_prefetch(&nw_n[i], _MM_HINT_T0);
    _mm_prefetch(&nh_n[i], _MM_HINT_T0);

    // only convert momentum and embedding, gradient is fp32
    __m256 gi = _mm256_loadu_ps(g + i);
    __m128i hhi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(h + i));
    __m256 hi = _mm256_cvtph_ps(hhi);
    __m128i whi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
    __m256 wi = _mm256_cvtph_ps(whi);

    __m256 nhi = _mm256_add_ps(hi, _mm256_mul_ps(gi, gi));
    __m128i nhhi = _mm256_cvtps_ph(nhi, 0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(nh + i), nhhi);

    __m256 vtmp = _mm256_div_ps(
        gi, _mm256_add_ps(_mm256_sqrt_ps(nhi), _mm256_set1_ps(epsilon)));
    __m256 nwi = _mm256_add_ps(wi, _mm256_mul_ps(_mm256_set1_ps(lr), vtmp));
    __m128i nhwi = _mm256_cvtps_ph(nwi, 0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(nw + i), nhwi);
  }

  for (; i < N; ++i) {
    float gi = g[i];
    float hi = h[i] + gi * gi;
    nh[i] = hi;
    nw[i] = w[i] + lr * gi / (std::sqrt(hi) + epsilon);
  }
}

void rowwise_adagrad_update__avx_f16c(
    int N,
    float* w,
    float* w_n, // prefetch ptr

    const float* g,

    float* h,
    float* h_n, // prefetch ptr

    float epsilon,
    float lr) {
  _mm_prefetch(h_n, _MM_HINT_T0);

  constexpr size_t kSize = 8;
  auto i = 0;

  __m256 partial_sum = _mm256_setzero_ps();
  for (; i + kSize <= N; i += kSize) {
    __m256 gi = _mm256_loadu_ps(g + i);
    partial_sum = _mm256_add_ps(partial_sum, _mm256_mul_ps(gi, gi));
  }
  // Reduce sum to 1 value
  __m256 partial_sum_2 = _mm256_hadd_ps(partial_sum, partial_sum);
  __m256 partial_sum_3 = _mm256_hadd_ps(partial_sum_2, partial_sum_2);
  float final_sum = _mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3)) +
      _mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1));

  for (; i < N; ++i) {
    final_sum += g[i] * g[i];
  }
  final_sum /= N;

  float hi = *h = *h + final_sum;
  float float_step = lr / (std::sqrt(hi) + epsilon);

  __m256 step = _mm256_set1_ps(float_step);

  for (i = 0; i + kSize <= N; i += kSize) {
    _mm_prefetch(&w_n[i], _MM_HINT_T0);

    __m256 gi = _mm256_loadu_ps(g + i);
    __m256 wi = _mm256_loadu_ps(w + i);

    _mm256_storeu_ps(w + i, _mm256_add_ps(wi, _mm256_mul_ps(gi, step)));
  }

  for (; i < N; ++i) {
    float gi = g[i];
    w[i] = w[i] + gi * float_step;
  }
}

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
    float lr) {
  constexpr size_t kSize = 8;
  auto i = 0;
  for (; i + kSize <= N; i += kSize) {
    __m256 gi = _mm256_loadu_ps(g + i);
    __m256 hi = _mm256_loadu_ps(h + i);
    __m256 wi = _mm256_loadu_ps(w + i);

    __m256 nhi = _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(decay), hi), _mm256_mul_ps(gi, gi));
    _mm256_storeu_ps(nh + i, nhi);
    __m256 vtmp = _mm256_div_ps(
        gi, _mm256_add_ps(_mm256_sqrt_ps(nhi), _mm256_set1_ps(epsilon)));
    _mm256_storeu_ps(
        nw + i, _mm256_add_ps(wi, _mm256_mul_ps(_mm256_set1_ps(lr), vtmp)));
  }

  for (; i < N; ++i) {
    float gi = g[i];
    float hi = nh[i] = decay * h[i] + gi * gi;
    nw[i] = w[i] + lr * gi / (std::sqrt(hi) + epsilon);
  }
}

SPARSE_ADAGRAD_SPECIALIZATION(int32_t, avx_f16c);
SPARSE_ADAGRAD_SPECIALIZATION(int64_t, avx_f16c);

} // namespace caffe2
