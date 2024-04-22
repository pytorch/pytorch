#include <algorithm>
#include <cstdint>

#include <immintrin.h>

namespace caffe2 {

namespace internal {

template <typename T>
void ReluAVX2(const int N, const int zero_point, const T* X, T* Y);

template <>
void ReluAVX2<uint8_t>(
    const int N,
    const int zero_point,
    const uint8_t* X,
    uint8_t* Y) {
  constexpr int kVLen = 32;
  const int n = N / kVLen * kVLen;
  const int r = N % kVLen;
  const __m256i zero_v = _mm256_set1_epi8(static_cast<uint8_t>(zero_point));
  for (int i = 0; i < n; i += kVLen) {
    __m256i cur_v = _mm256_max_epu8(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X + i)), zero_v);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Y + i), cur_v);
  }
  for (int i = 0; i < r; ++i) {
    Y[n + i] = std::max(X[n + i], static_cast<uint8_t>(zero_point));
  }
}

template <>
void ReluAVX2<uint16_t>(
    const int N,
    const int zero_point,
    const uint16_t* X,
    uint16_t* Y) {
  constexpr int kVLen = 16;
  const int n = N / kVLen * kVLen;
  const int r = N % kVLen;
  const __m256i zero_v = _mm256_set1_epi16(static_cast<uint16_t>(zero_point));
  for (int i = 0; i < n; i += kVLen) {
    __m256i cur_v = _mm256_max_epu16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X + i)), zero_v);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Y + i), cur_v);
  }
  for (int i = 0; i < r; ++i) {
    Y[n + i] = std::max(X[n + i], static_cast<uint16_t>(zero_point));
  }
}

} // namespace internal

} // namespace caffe2
