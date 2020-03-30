#include "caffe2/perfkernels/adagrad.h"
#include "caffe2/perfkernels/cvtsh_ss_bugfix.h"

#include <emmintrin.h>
#include <immintrin.h>

namespace caffe2 {

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
        _mm256_mul_ps(_mm256_set1_ps(lr), gi),
        _mm256_add_ps(_mm256_sqrt_ps(nhi), _mm256_set1_ps(epsilon)));
    _mm256_storeu_ps(nw + i, _mm256_add_ps(wi, vtmp));
  }

  for (; i < N; ++i) {
    float gi = g[i];
    float hi = nh[i] = decay * h[i] + gi * gi;
    nw[i] = w[i] + lr * gi / (std::sqrt(hi) + epsilon);
  }
}

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
  internal::adagrad_update_prefetch_inlined(
      N, w, w_n, g, h, h_n, nw, nw_n, nh, nh_n, epsilon, lr);
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
  constexpr int kSize = 8;
  auto i = 0;
  for (; i + kSize <= N; i += kSize) {
    _mm_prefetch(reinterpret_cast<const char*>(&w_n[i]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&h_n[i]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&nw_n[i]), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(&nh_n[i]), _MM_HINT_T0);

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
        _mm256_mul_ps(_mm256_set1_ps(lr), gi),
        _mm256_add_ps(_mm256_sqrt_ps(nhi), _mm256_set1_ps(epsilon)));
    __m256 nwi = _mm256_add_ps(wi, vtmp);
    __m128i nhwi = _mm256_cvtps_ph(nwi, 0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(nw + i), nhwi);
  }

  for (; i < N; ++i) {
    float gi = g[i];
    float nhi =
        _cvtsh_ss(reinterpret_cast<const unsigned short*>(h)[i]) + gi * gi;
    reinterpret_cast<unsigned short*>(nh)[i] = _cvtss_sh(nhi, 0);
    float nwi = _cvtsh_ss(reinterpret_cast<const unsigned short*>(w)[i]) +
        lr * gi / (std::sqrt(nhi) + epsilon);
    reinterpret_cast<unsigned short*>(nw)[i] = _cvtss_sh(nwi, 0);
  }
}

} // namespace caffe2
