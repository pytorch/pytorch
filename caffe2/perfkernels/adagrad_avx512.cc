#include "caffe2/perfkernels/adagrad.h"
#include "caffe2/perfkernels/cvtsh_ss_bugfix.h"

#include <emmintrin.h>
#include <immintrin.h>

namespace caffe2 {

// version without prefetching
void adagrad_update__avx512(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    float decay,
    float lr,
    float weight_decay = 0.f) {
  constexpr int kSize = 16;
  auto i = 0;
  for (; i + kSize <= N; i += kSize) {
    __m512 gi = _mm512_loadu_ps(g + i);
    __m512 hi = _mm512_loadu_ps(h + i);
    __m512 wi = _mm512_loadu_ps(w + i);
    gi = _mm512_fmadd_ps(_mm512_set1_ps(weight_decay), wi, gi);

    __m512 nhi = _mm512_add_ps(
        _mm512_mul_ps(_mm512_set1_ps(decay), hi), _mm512_mul_ps(gi, gi));
    _mm512_storeu_ps(nh + i, nhi);
    __m512 vtmp = _mm512_div_ps(
        _mm512_mul_ps(_mm512_set1_ps(lr), gi),
        _mm512_add_ps(_mm512_sqrt_ps(nhi), _mm512_set1_ps(epsilon)));
    _mm512_storeu_ps(nw + i, _mm512_add_ps(wi, vtmp));
  }

  for (; i < N; ++i) {
    float gi = std::fma(weight_decay, w[i], g[i]);
    float hi = nh[i] = decay * h[i] + gi * gi;
    nw[i] = w[i] + lr * gi / (std::sqrt(hi) + epsilon);
  }
}

} // namespace caffe2
