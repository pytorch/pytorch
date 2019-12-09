#include <cstdint>
#include <limits>

#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <fbgemm/QuantUtils.h>

namespace caffe2 {

namespace internal {

template <typename T>
void SegmentMomentsAVX2(
    const int N,
    const T* src,
    int64_t* sum,
    int64_t* sumsq);

template <>
void SegmentMomentsAVX2<uint8_t>(
    const int N,
    const uint8_t* src,
    int64_t* sum,
    int64_t* sumsq) {
  constexpr int kVLen = 16;
  const int n = N / kVLen * kVLen;
  const int r = N % kVLen;
  const __m256i kOneInt16 = _mm256_set1_epi16(0x01);
  __m256i sum_v = _mm256_setzero_si256();
  __m256i sumsq_v = _mm256_setzero_si256();
  for (int i = 0; i < n; i += kVLen) {
    const __m256i cur_v = _mm256_cvtepu8_epi16(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i)));
    sum_v = _mm256_add_epi32(sum_v, _mm256_madd_epi16(cur_v, kOneInt16));
    sumsq_v = _mm256_add_epi32(sumsq_v, _mm256_madd_epi16(cur_v, cur_v));
  }
  int32_t sum_arr[8];
  int32_t sumsq_arr[8];
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(sum_arr), sum_v);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(sumsq_arr), sumsq_v);
  for (int i = 0; i < 8; ++i) {
    *sum += static_cast<int64_t>(sum_arr[i]);
    *sumsq += static_cast<int64_t>(sumsq_arr[i]);
  }
  for (int i = 0; i < r; ++i) {
    *sum += static_cast<int64_t>(src[n + i]);
    *sumsq +=
        static_cast<int64_t>(src[n + i]) * static_cast<int64_t>(src[n + i]);
  }
}

template <typename T>
void VectorMomentsAVX2(const int N, const T* src, int64_t* sum, int64_t* sumsq);

template <>
void VectorMomentsAVX2<uint8_t>(
    const int N,
    const uint8_t* src,
    int64_t* sum,
    int64_t* sumsq) {
  constexpr int kVLen = 32768;
  const int n = N / kVLen * kVLen;
  const int r = N % kVLen;
  for (int i = 0; i < n; i += kVLen) {
    SegmentMomentsAVX2<uint8_t>(kVLen, src + i, sum, sumsq);
  }
  if (r > 0) {
    SegmentMomentsAVX2<uint8_t>(r, src + n, sum, sumsq);
  }
}

void ComputeQuantizedFusedParamsAVX2(
    const int N,
    const int G,
    const int K,
    const int32_t X_zero_point,
    const int32_t* mu,
    const int32_t* rsig,
    const int32_t* gamma,
    int32_t* scale,
    int32_t* bias) {
  constexpr int kVLen = 8;
  const int k = K / kVLen * kVLen;
  const int r = K % kVLen;
  for (int n = N - 1; n >= 0; --n) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int g = 0; g < G; ++g) {
      const __m256i mu_v = _mm256_set1_epi32(mu[n * G + g] + X_zero_point);
      const __m256i rsig_v = _mm256_set1_epi32(rsig[n * G + g]);
      for (int i = 0; i < k; i += kVLen) {
        const __m256i gamma_v =
            _mm256_loadu_si256((const __m256i*)(gamma + g * K + i));
        const __m256i beta_v =
            _mm256_loadu_si256((const __m256i*)(bias + g * K + i));
        __m256i scale_v = _mm256_mullo_epi32(gamma_v, rsig_v);
        __m256i bias_v =
            _mm256_sub_epi32(beta_v, _mm256_mullo_epi32(scale_v, mu_v));
        const int offset = (n * G + g) * K + i;
        _mm256_storeu_si256((__m256i*)(scale + offset), scale_v);
        _mm256_storeu_si256((__m256i*)(bias + offset), bias_v);
      }
      for (int i = 0; i < r; ++i) {
        const int offset = (n * G + g) * K + k + i;
        scale[offset] = gamma[g * K + k + i] * rsig[n * G + g];
        bias[offset] = bias[g * K + k + i] -
            scale[offset] * (mu[n * G + g] + X_zero_point);
      }
    }
  }
}

#define INIT_REQUANTIZE_AVX2                                      \
  const __m256i b = _mm256_set1_epi32(params.multiplier);         \
  const __m256i prev_shift_nudge = _mm256_set1_epi64x(            \
      (1ll << (params.right_shift - 1)) + 0x8000000000000000ULL); \
  const __m256i post_shift_nudge = _mm256_set1_epi64x(            \
      params.target_qparams.zero_point -                          \
      (0x8000000000000000ULL >> params.right_shift));             \
  const __m256i min_v =                                           \
      _mm256_set1_epi32(std::numeric_limits<uint8_t>::min());     \
  const __m256i max_v =                                           \
      _mm256_set1_epi32(std::numeric_limits<uint8_t>::max());     \
  const __m256i shuffle_mask_v = _mm256_set_epi8(                 \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0x0c,                                                       \
      0x08,                                                       \
      0x04,                                                       \
      0x00,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0xff,                                                       \
      0x0c,                                                       \
      0x08,                                                       \
      0x04,                                                       \
      0x00);                                                      \
  const __m256i permute_mask_v =                                  \
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

#define REQUANTIZE_AVX2(params, src, dst)                                     \
  do {                                                                        \
    __m256i a_v = (src);                                                      \
    __m256i a_even_v = a_v;                                                   \
    __m256i a_odd_v = _mm256_srli_si256(a_v, 4);                              \
    __m256i ab_even_v = _mm256_mul_epi32(a_even_v, b);                        \
    __m256i ab_odd_v = _mm256_mul_epi32(a_odd_v, b);                          \
    __m256i even_rounded_v = _mm256_add_epi64(ab_even_v, prev_shift_nudge);   \
    __m256i odd_rounded_v = _mm256_add_epi64(ab_odd_v, prev_shift_nudge);     \
    __m256i even_result_v = _mm256_add_epi64(                                 \
        _mm256_srli_epi64(even_rounded_v, params.right_shift),                \
        post_shift_nudge);                                                    \
    __m256i odd_result_v = _mm256_add_epi64(                                  \
        _mm256_srli_epi64(odd_rounded_v, params.right_shift),                 \
        post_shift_nudge);                                                    \
    odd_result_v = _mm256_slli_si256(odd_result_v, 4);                        \
    __m256i result_v = _mm256_blend_epi32(even_result_v, odd_result_v, 0xaa); \
    __m256i clipped_v =                                                       \
        _mm256_max_epi32(min_v, _mm256_min_epi32(max_v, result_v));           \
    clipped_v = _mm256_shuffle_epi8(clipped_v, shuffle_mask_v);               \
    clipped_v = _mm256_permutevar8x32_epi32(clipped_v, permute_mask_v);       \
    *(int64_t*)(dst) = _mm256_extract_epi64(clipped_v, 0);                    \
  } while (false)

template <typename T>
void AffineBatchChannelAndRequantizeNCHWAVX2(
    const int N,
    const int C,
    const int HxW,
    const fbgemm::RequantizationParams& params,
    const T* X,
    const int32_t* scale,
    const int32_t* bias,
    T* Y);

template <>
void AffineBatchChannelAndRequantizeNCHWAVX2<uint8_t>(
    const int N,
    const int C,
    const int HxW,
    const fbgemm::RequantizationParams& params,
    const uint8_t* X,
    const int32_t* scale,
    const int32_t* bias,
    uint8_t* Y) {
  INIT_REQUANTIZE_AVX2;
  constexpr int kVLen = 8;
  const int outer_size = N * C;
  const int n = HxW / kVLen * kVLen;
  const int r = HxW % kVLen;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < outer_size; ++i) {
    const uint8_t* X_ptr = X + i * HxW;
    uint8_t* Y_ptr = Y + i * HxW;
    const __m256i scale_v = _mm256_set1_epi32(scale[i]);
    const __m256i bias_v = _mm256_set1_epi32(bias[i]);
    for (int j = 0; j < n; j += kVLen) {
      const __m256i cur_v =
          _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(X_ptr + j)));
      REQUANTIZE_AVX2(
          params,
          _mm256_add_epi32(_mm256_mullo_epi32(cur_v, scale_v), bias_v),
          (Y_ptr + j));
    }
    for (int j = 0; j < r; ++j) {
      Y_ptr[n + j] = fbgemm::Requantize<uint8_t>(
          static_cast<int32_t>(X_ptr[n + j]) * scale[i] + bias[i], params);
    }
  }
}

template <typename T>
void AffineBatchChannelAndRequantizeNHWCAVX2(
    const int N,
    const int C,
    const int HxW,
    const fbgemm::RequantizationParams& params,
    const T* X,
    const int32_t* scale,
    const int32_t* bias,
    T* Y);

template <>
void AffineBatchChannelAndRequantizeNHWCAVX2<uint8_t>(
    const int N,
    const int C,
    const int HxW,
    const fbgemm::RequantizationParams& params,
    const uint8_t* X,
    const int32_t* scale,
    const int32_t* bias,
    uint8_t* Y) {
  INIT_REQUANTIZE_AVX2;
  constexpr int kVLen = 8;
  const int outer_size = N * HxW;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < outer_size; ++i) {
    const int c = i / HxW * C;
    const int n = C / kVLen * kVLen;
    const int r = C % kVLen;
    const uint8_t* X_ptr = X + i * C;
    uint8_t* Y_ptr = Y + i * C;
    for (int j = 0; j < n; j += kVLen) {
      const __m256i cur_v =
          _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(X_ptr + j)));
      const __m256i scale_v =
          _mm256_loadu_si256((const __m256i*)(scale + c + j));
      const __m256i bias_v = _mm256_loadu_si256((const __m256i*)(bias + c + j));
      REQUANTIZE_AVX2(
          params,
          _mm256_add_epi32(_mm256_mullo_epi32(cur_v, scale_v), bias_v),
          (Y_ptr + j));
    }
    for (int j = 0; j < r; ++j) {
      Y_ptr[n + j] = fbgemm::Requantize<uint8_t>(
          static_cast<int32_t>(X_ptr[n + j]) * scale[c + n + j] +
              bias[c + n + j],
          params);
    }
  }
}

#undef REQUANTIZE_AVX2
#undef INIT_REQUANTIZE_AVX2

} // namespace internal

} // namespace caffe2
