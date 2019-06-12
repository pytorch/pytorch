#include "dnnlowp.h"
#include "caffe2/core/logging.h"
#include "dnnlowp_op.h"
#include "kl_minimization.h"
#include "l2_minimization.h"

#include <cassert>
#include <cctype>
#ifdef _OPENMP
#include <omp.h>
#endif

C10_DEFINE_int32(
  dnnlowp_activation_quantization_precision, 8,
  "Precision used for activation tensors");
C10_DEFINE_int32(
  dnnlowp_weight_quantization_precision, 8,
  "Precision used for weight tensors");
C10_DEFINE_int32(
  dnnlowp_requantization_multiplier_precision, 32,
  "Precision of integer multipliers used for rescaling quantized numbers");
C10_DEFINE_int32(
  dnnlowp_eltwise_quantization_precision, 16,
  "Precision used for intermediate numbers during elementwise operations");
C10_DEFINE_bool(
  dnnlowp_force_scale_power_of_two, false,
  "When true, force quantization scales to a power of two");
C10_DEFINE_bool(
  dnnlowp_preserve_activation_sparsity, false,
  "When true, 0 is mapped to 0 after quantization: "
  "i.e., symmetric quantization");
C10_DEFINE_bool(
  dnnlowp_preserve_weight_sparsity, false,
  "When true, 0 is mapped to 0 after quantization: "
  "i.e., symmetric quantization");
C10_DEFINE_string(
  dnnlowp_activation_quantization_kind, "min_max",
  "Quantization method for activation tensors. "
  "Allowed values: min_max, l2, l2_approx, kl, l1, p99");
C10_DEFINE_string(
  dnnlowp_weight_quantization_kind, "min_max",
  "Quantization method for weight tensors. "
  "Allowed values: min_max, l2, l2_approx, kl, l1, p99");
C10_DEFINE_int32(
  dnnlowp_nbits_in_non_outlier, 8,
  "When outlier-aware quantization is used, if a quantized number can be "
  "represented by this number of bits, it is considered not an outlier so "
  "handled with 16-bit accumulation");
C10_DEFINE_int32(
  dnnlowp_copy_to_32bit_frequency, 32,
  "When outlier-aware quantization is used, this option specifies how often "
  "we spill 16-bit accumulated numbers to 32-bit during the first pass");

namespace dnnlowp {

using namespace std;

float TensorQuantizationParams::Min() const {
  return Dequantize(0, *this);
}

float TensorQuantizationParams::Max() const {
  return Dequantize((1 << precision) - 1, *this);
}

int64_t SaturatingRoundingMulWithShift(int32_t a, int32_t b, int right_shift) {
  int64_t a_64(a);
  int64_t b_64(b);
  int64_t ab_64 = a_64 * b_64;

  int64_t nudge = 1ll << (right_shift - 1);
  return (ab_64 + nudge) >> right_shift;
}

#ifdef __AVX2__
void RequantizeFixedPointAvx2(
    const int32_t *src, uint8_t *dst, int len,
    const RequantizationParams& params) {
  constexpr int VLEN = 8;

  __m256i b = _mm256_set1_epi32(params.multiplier);

  // AVX2 doesn't support arithmetic right shift.
  // As a work around, we convert 64-bit multiplied results to uint64_t by
  // adding 0x8000000000000000ULL, logical right shift, and subtract by
  // (0x8000000000000000ULL >> right_shift).
  __m256i pre_shift_nudge = _mm256_set1_epi64x(
      (1ll << (params.right_shift - 1)) + 0x8000000000000000ULL);
  __m256i post_shift_nudge = _mm256_set1_epi64x(
      params.target_qparams.zero_point -
      (0x8000000000000000ULL >> params.right_shift));

  __m256i min_v = _mm256_set1_epi32(numeric_limits<uint8_t>::min());
  __m256i max_v = _mm256_set1_epi32(numeric_limits<uint8_t>::max());

  __m256i shuffle_mask_v = _mm256_set_epi8(
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0x0c, 0x08, 0x04, 0x00,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0x0c, 0x08, 0x04, 0x00);
  __m256i permute_mask_v = _mm256_set_epi32(
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);

  int i = 0;
  for ( ; i < len / VLEN * VLEN; i += VLEN) {
    __m256i a_v = _mm256_loadu_si256((const __m256i *)(src + i));

    // a = a0 | a1 | a2 | a3 | a4 | a5 | a6 | a7
    // b = b0 | b1 | b3 | b3 | b4 | b5 | b6 | b7
    __m256i a_even_v = a_v;
    __m256i a_odd_v = _mm256_srli_si256(a_v, 4);

    __m256i ab_even_v = _mm256_mul_epi32(a_even_v, b);
    __m256i ab_odd_v = _mm256_mul_epi32(a_odd_v, b);

    __m256i even_rounded_v = _mm256_add_epi64(ab_even_v, pre_shift_nudge);
    __m256i odd_rounded_v = _mm256_add_epi64(ab_odd_v, pre_shift_nudge);

    __m256i even_result_v = _mm256_add_epi64(
        _mm256_srli_epi64(even_rounded_v, params.right_shift),
        post_shift_nudge);
    __m256i odd_result_v = _mm256_add_epi64(
        _mm256_srli_epi64(odd_rounded_v, params.right_shift),
        post_shift_nudge);
    odd_result_v = _mm256_slli_si256(odd_result_v, 4);

    // even_result_v has numbers we want in its even 32-bit SIMD lanes, and
    // odd_result_v has numbers we want in its odd 32-bit SIMD lanes.
    // Use blend to combine them.
    __m256i result_v = _mm256_blend_epi32(even_result_v, odd_result_v, 0xaa);
    __m256i clipped_v = _mm256_max_epi32(
      min_v, _mm256_min_epi32(max_v, result_v));

    clipped_v = _mm256_shuffle_epi8(clipped_v, shuffle_mask_v);
    clipped_v = _mm256_permutevar8x32_epi32(clipped_v, permute_mask_v);
    *(int64_t *)(dst + i) = _mm256_extract_epi64(clipped_v, 0);
  }

  for ( ; i < len; ++i) {
    dst[i] = RequantizeFixedPoint<uint8_t>(src[i], params);
  }
}

void RequantizeAvx2(
    const int32_t *src, uint8_t *dst, int len,
    const RequantizationParams& params) {
  // Adoption of implementation at QNNPACK/src/requantization/fp32-sse2.c
  // using AVX2 instructions
  constexpr int VLEN = 8;

  __m256 multiplier_v = _mm256_set1_ps(params.real_multiplier);
  __m256i zero_point_v = _mm256_set1_epi16(params.target_qparams.zero_point);

  __m256i min_v = _mm256_set1_epi8(numeric_limits<uint8_t>::min());
  __m256i max_v = _mm256_set1_epi8(numeric_limits<uint8_t>::max());

  __m256i permute_mask_v =
      _mm256_set_epi32(0x07, 0x03, 0x06, 0x02, 0x05, 0x01, 0x04, 0x00);

  int i = 0;
  for ( ; i < len / (VLEN * 4) * (VLEN * 4); i += (VLEN * 4)) {
    __m256i x_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    __m256i y_v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i + VLEN));
    __m256i z_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(src + i + 2 * VLEN));
    __m256i w_v = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(src + i + 3 * VLEN));

    /*
     * Convert int32_t input to FP32 and multiply by FP32 scale.
     * Both operations involve statistically unbiased roundings (with default
     * MXCSR rounding mode):
     * - Large int32_t values can't be exactly represented as FP32. CVTDQ2PS
     * instruction on x86 would round it according to nearest FP32 value with
     * ties to even (assuming default MXCSR rounding mode).
     * - Product of two FP32 values is generally not exactly representation as
     * an FP32 value, and will be rounded to nearest FP32 value with ties to
     * even with default MXCSR rounding mode.
     */
    __m256 x_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(x_v), multiplier_v);
    __m256 y_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(y_v), multiplier_v);
    __m256 z_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(z_v), multiplier_v);
    __m256 w_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(w_v), multiplier_v);

    /*
     * Convert scaled FP32 result to int32_t using CVTPS2DQ instruction.
     * CVTPS2DQ instruction rounds result according to nearest FP32 value with
     * ties to even (assuming default MXCSR rounding mode). However, when
     * conversion overflows, it produces INT32_MIN as a result. For large
     * positive inputs the result of conversion can become negative, which
     * affects the final requantization result. Note that on x86 SSE2 we have
     * e.g. int32_t(float(INT32_MAX)) == INT32_MIN! This happens because
     * float(INT32_MAX) rounds to 2**31, which overflows int32_t when it is
     * converted back to integer.
     *
     * Thankfully, we can prove that overflow never happens in this
     * requantization scheme. The largest positive input is INT32_MAX (2**31 -
     * 1), which turns into 2**31 when converted to float. The largest scale
     * value is 0x1.FFFFFEp-1. When multiplied together, the result is
     * 2147483520 (compare to INT32_MAX = 2147483647), which fits into int32_t
     * without overflow.
     */
    __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);
    __m256i y_rounded_v = _mm256_cvtps_epi32(y_scaled_v);
    __m256i z_rounded_v = _mm256_cvtps_epi32(z_scaled_v);
    __m256i w_rounded_v = _mm256_cvtps_epi32(w_scaled_v);

    /*
     * Standard final sequence on x86 AVX2:
     * - Pack to int16_t and saturate
     * - Add zero point
     * - Pack to uint8_t and saturate
     * - Clamp between qmin and qmax
     */
    __m256i xy_packed_v = _mm256_adds_epi16(
        _mm256_packs_epi32(x_rounded_v, y_rounded_v), zero_point_v);
    __m256i zw_packed_v = _mm256_adds_epi16(
        _mm256_packs_epi32(z_rounded_v, w_rounded_v), zero_point_v);
    __m256i xyzw_packed_v = _mm256_packus_epi16(xy_packed_v, zw_packed_v);
    __m256i xyzw_clamped_v =
        _mm256_max_epu8(min_v, _mm256_min_epu8(xyzw_packed_v, max_v));

    /*
     * xyzw_clamped_v has results in the following layout so we need to permute:
     * x0-3 y0-3 z0-3 w0-3 x4-7 y4-7 z4-7 w4-7
     */
    xyzw_clamped_v =
        _mm256_permutevar8x32_epi32(xyzw_clamped_v, permute_mask_v);

    /*
     * 4x CVTDQ2PS
     * 4x MULPS
     * 4x CVTPS2DQ
     * 2x PACKSSDW
     * 1x PACKUSWB
     * 2x PADDW
     * 1x PMAXUB
     * 1x PMINUB
     * 1x PERMD
     * ---------------------
     * 20 instructions total
     */
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), xyzw_clamped_v);
  } // i loop vectorized and unrolled 4x

  for ( ; i < len / VLEN * VLEN; i += VLEN) {
    __m256i x_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
    __m256 x_scaled_v = _mm256_mul_ps(_mm256_cvtepi32_ps(x_v), multiplier_v);
    __m256i x_rounded_v = _mm256_cvtps_epi32(x_scaled_v);
    __m256i x_packed_v = _mm256_adds_epi16(
        _mm256_packs_epi32(x_rounded_v, _mm256_setzero_si256()), zero_point_v);
    x_packed_v = _mm256_packus_epi16(x_packed_v, _mm256_setzero_si256());
    __m256i x_clamped_v =
        _mm256_max_epu8(min_v, _mm256_min_epu8(x_packed_v, max_v));

    /*
     * x_clamped_v has results in the following layout so we need to permute:
     * x0-3 garbage0-11 x4-7 garbage12-23
     */
    x_clamped_v =
        _mm256_permutevar8x32_epi32(x_clamped_v, permute_mask_v);

    /*
     * 1x CVTDQ2PS
     * 1x MULPS
     * 1x CVTPS2DQ
     * 1x PACKSSDW
     * 1x PACKUSWB
     * 1x PADDW
     * 1x PMAXUB
     * 1x PMINUB
     * 1x PERMD
     * ---------------------
     * 9 instructions total
     */
    _mm_storel_epi64(
        reinterpret_cast<__m128i*>(dst + i),
        _mm256_castsi256_si128(x_clamped_v));
  } // i loop vectorized

  for ( ; i < len; ++i) {
    dst[i] = Requantize<uint8_t>(src[i], params);
  } // i loop remainder
}
#endif

#define DNNLOWP_SPECIALIZED_REQUANTIZE(T)     \
  template <>                                 \
  void Requantize<T>(                         \
      const int32_t* src,                     \
      T* dst,                                 \
      const int len,                          \
      const RequantizationParams& params) {   \
    for (int i = 0; i < len; ++i) {           \
      dst[i] = Requantize<T>(src[i], params); \
    }                                         \
  }
DNNLOWP_SPECIALIZED_REQUANTIZE(uint16_t)
DNNLOWP_SPECIALIZED_REQUANTIZE(int32_t)
#undef DNNLOWP_SPECIALIZED_REQUANTIZE

template <>
void Requantize<uint8_t>(
    const int32_t* src,
    uint8_t* dst,
    const int len,
    const RequantizationParams& params) {
  if (params.target_qparams.precision == 8 && caffe2::GetCpuId().avx2()) {
    RequantizeAvx2(src, dst, len, params);
  } else {
    for (int i = 0; i < len; ++i) {
      dst[i] = Requantize<uint8_t>(src[i], params);
    }
  }
}

template <typename T>
void Quantize(
    const float* src,
    T* dst,
    int len,
    const TensorQuantizationParams& qparams) {

#if defined(__AVX2__) && defined(__FMA__)
  caffe2::CpuId cpuid = caffe2::GetCpuId();
  bool avx2_support = cpuid.avx2();
  bool fma_support = cpuid.fma();
  if (avx2_support && fma_support && qparams.precision == 8 &&
      std::is_same<T, uint8_t>::value) {
    // fast path
    constexpr int VLEN = 8;
    std::size_t i = 0;
    __m256 inverse_scale_v = _mm256_set1_ps(1.f / qparams.scale);
    for (; i < len / VLEN * VLEN; i += VLEN) {
      __m256 src_v = _mm256_loadu_ps(src + i);
      __m256 transformed_v = _mm256_fmadd_ps(
          src_v, inverse_scale_v, _mm256_set1_ps(qparams.zero_point));
      __m256 clipped_v = _mm256_min_ps(
          _mm256_max_ps(transformed_v, _mm256_set1_ps(0.f)),
          _mm256_set1_ps(255.f));
      __m256i rounded_v = _mm256_cvtps_epi32(clipped_v);
      std::int32_t temp_int32[VLEN] __attribute__((aligned(64)));
      _mm256_store_si256((__m256i*)temp_int32, rounded_v);
      for (int j = 0; j < VLEN; ++j) {
        dst[i + j] = temp_int32[j];
      }
    }

    for (; i < len; ++i) {
      float transformed = qparams.zero_point + src[i] / qparams.scale;
      float clipped = std::min(std::max(transformed, 0.f), 255.f);
      dst[i] = round(clipped);
    }
  } else
#endif
  {
    for (std::size_t i = 0; i < len; ++i) {
      dst[i] = dnnlowp::Quantize<T>(src[i], qparams);
    }
  }
}

template void Quantize<uint8_t>(
    const float* src,
    uint8_t* dst,
    int len,
    const TensorQuantizationParams& qparams);

template void Quantize<int8_t>(
    const float* src,
    int8_t* dst,
    int len,
    const TensorQuantizationParams& qparams);

template void Quantize<uint16_t>(
    const float* src,
    uint16_t* dst,
    int len,
    const TensorQuantizationParams& qparams);

template void Quantize<int16_t>(
    const float* src,
    int16_t* dst,
    int len,
    const TensorQuantizationParams& qparams);

QuantizationFactory::QuantizationKind StringToKind(const string& s) {
  string s_lower(s);
  transform(s_lower.begin(), s_lower.end(), s_lower.begin(), ::tolower);

  if (s_lower == "min_max") {
    return QuantizationFactory::MIN_MAX_QUANTIZATION;
  } else if (s_lower == "l1") {
    return QuantizationFactory::L1_MIN_QUANTIZATION;
  } else if (s_lower == "l2") {
    return QuantizationFactory::L2_MIN_QUANTIZATION;
  } else if (s_lower == "l2_approx") {
    if (FLAGS_dnnlowp_preserve_weight_sparsity ||
        FLAGS_dnnlowp_preserve_activation_sparsity) {
      return QuantizationFactory::L2_MIN_QUANTIZATION;
    } else {
      return QuantizationFactory::L2_MIN_QUANTIZATION_APPROX;
    }
  } else if (s_lower == "kl") {
    return QuantizationFactory::KL_MIN_QUANTIZATION;
  } else if (s_lower == "p99") {
    return QuantizationFactory::P99_QUANTIZATION;
  } else {
    assert(false);
    return QuantizationFactory::MIN_MAX_QUANTIZATION;
  }
}

QuantizationFactory *QuantizationFactory::GetDefaultInstance() {
  static QuantizationFactory singleton(
    FLAGS_dnnlowp_activation_quantization_precision,
    FLAGS_dnnlowp_weight_quantization_precision,
    FLAGS_dnnlowp_requantization_multiplier_precision,
    FLAGS_dnnlowp_eltwise_quantization_precision,
    FLAGS_dnnlowp_preserve_activation_sparsity,
    FLAGS_dnnlowp_preserve_weight_sparsity,
    FLAGS_dnnlowp_force_scale_power_of_two,
    StringToKind(FLAGS_dnnlowp_activation_quantization_kind),
    StringToKind(FLAGS_dnnlowp_weight_quantization_kind));

  static bool log_printed = false;
  if (!log_printed) {
    LOG(INFO) <<
      "activation_precision " <<
      FLAGS_dnnlowp_activation_quantization_precision;
    LOG(INFO) <<
      "weight_precision " << FLAGS_dnnlowp_weight_quantization_precision;
    LOG(INFO) <<
      "requantization_multiplier_precision " <<
      FLAGS_dnnlowp_requantization_multiplier_precision;
    LOG(INFO) <<
      "eltwise_quantize_precision " <<
      FLAGS_dnnlowp_eltwise_quantization_precision;
    LOG(INFO) <<
      "preserve_activation_sparsity " <<
      FLAGS_dnnlowp_preserve_activation_sparsity;
    LOG(INFO) <<
      "preserve_weight_sparsity " << FLAGS_dnnlowp_preserve_weight_sparsity;
    LOG(INFO) <<
      "force_scale_power_of_two " << FLAGS_dnnlowp_force_scale_power_of_two;
    LOG(INFO) <<
      "activation_quantization_kind " <<
      FLAGS_dnnlowp_activation_quantization_kind;
    LOG(INFO) <<
      "weight_quantization_kind " << FLAGS_dnnlowp_weight_quantization_kind;
    LOG(INFO) << "nbits_in_non_outlier " << FLAGS_dnnlowp_nbits_in_non_outlier;
    LOG(INFO) <<
      "copy_to_32bit_frequency " << FLAGS_dnnlowp_copy_to_32bit_frequency;
    LOG(INFO) << "omp_get_max_threads() " << caffe2::dnnlowp_get_max_threads();

    log_printed = true;
  }

  return &singleton;
}

QuantizationFactory::QuantizationFactory(
    int activation_precision,
    int weight_precision,
    int requantization_multiplier_precision,
    int eltwise_quantize_precision,
    bool preserve_activation_sparsity,
    bool preserve_weight_sparsity,
    bool force_scale_power_of_two,
    QuantizationKind activation_kind, QuantizationKind weight_kind) :
  activation_precision_(activation_precision),
  weight_precision_(weight_precision),
  requantization_multiplier_precision_(requantization_multiplier_precision),
  eltwise_quantize_precision_(eltwise_quantize_precision),
  preserve_activation_sparsity_(preserve_activation_sparsity),
  preserve_weight_sparsity_(preserve_weight_sparsity),
  force_scale_power_of_two_(force_scale_power_of_two),
  activation_kind_(activation_kind), weight_kind_(weight_kind) {
}

TensorQuantizationParams QuantizationFactory::ChooseQuantizationParams(
    const Histogram& hist, QuantizationKind kind,
    int precision, bool preserve_sparsity) const {
  switch (kind) {
  case L2_MIN_QUANTIZATION:
    return L2ErrorMinimization().ChooseQuantizationParams(
      hist, preserve_sparsity, precision);
  case L2_MIN_QUANTIZATION_APPROX:
    return L2ErrorMinimization().NonlinearQuantizationParamsSearch(
      hist, preserve_sparsity, precision);
  case L1_MIN_QUANTIZATION:
    return L1ErrorMinimization().ChooseQuantizationParams(
      hist, preserve_sparsity, precision);
  case KL_MIN_QUANTIZATION:
    return KLDivergenceMinimization().ChooseQuantizationParams(
      hist, preserve_sparsity, precision);
  case P99_QUANTIZATION:
    assert(preserve_sparsity);
    return P99().ChooseQuantizationParams(hist, preserve_sparsity, precision);
  case MIN_MAX_QUANTIZATION:
  default:
    return ChooseQuantizationParams(
        hist.Min(), hist.Max(), precision, preserve_sparsity);
  }
}

TensorQuantizationParams QuantizationFactory::ChooseQuantizationParams(
    const Histogram& hist, bool is_weight) const {
  if (is_weight) {
    return ChooseQuantizationParams(
        hist, GetWeightKind(), GetWeightPrecision(),
        GetPreserveWeightSparsity());
  }
  else {
    return ChooseQuantizationParams(
        hist, GetActivationKind(), GetActivationPrecision(),
        GetPreserveActivationSparsity());
  }
}

TensorQuantizationParams QuantizationFactory::ChooseQuantizationParams_(
    float min, float max,
    int32_t qmin, int32_t qmax, bool preserve_sparsity) const {

  if (min < 0 && max > 0 && preserve_sparsity) {
    int symmetric_qmin = -((qmax - qmin) / 2 + 1);
    int symmetric_qmax = (qmax - qmin) / 2;
    double max_scale =
      std::max(fabs(min / symmetric_qmin), fabs(max / symmetric_qmax));
    min = max_scale * symmetric_qmin;
    max = max_scale * symmetric_qmax;
  }

  double scale =
    (std::max(max, 0.f) - std::min(min, 0.f)) / ((double)qmax - qmin);
  if (scale == 0) scale = 0.1;
    // If scale is 0, we arbitrary adjust the scale to 0.1
  assert(scale > 0);

  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  if (force_scale_power_of_two_) {
    if (scale < 1) {
      scale = 1./(1 << (int)floor(log2(1/scale)));
    }
    else {
      scale = 1 << (int)ceil(log2(scale));
    }
  }

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  double zero_point_from_min = qmin - min / scale;
  double zero_point_from_max = qmax - max / scale;
  double zero_point_from_min_error = std::abs(qmin) + std::abs(min / scale);
  double zero_point_from_max_error = std::abs(qmax) + std::abs(max / scale);
  double initial_zero_point =
    zero_point_from_min_error < zero_point_from_max_error
    ? zero_point_from_min : zero_point_from_max;

  // for symmetric quantization (min == -max), we force zero_point to 128
  // to model signed integer (FIXME: this is a workaround that gemmlowp
  // doesn't support signed int AFAIK. Once we have an (efficient) gemm for
  // signed as well, we can just use signed int with zero_point = 0
  if (min < 0 && max > 0 && preserve_sparsity) {
    initial_zero_point = (qmin + qmax) / 2 + 1;
  }

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  // padding).
  int32_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = static_cast<int32_t>(round(initial_zero_point));
  }

  TensorQuantizationParams result;
  result.scale = scale;
  result.zero_point = nudged_zero_point;
  return result;
}

TensorQuantizationParams QuantizationFactory::ChooseQuantizationParams(
    const float *values, int len, QuantizationKind kind,
    int precision, bool preserve_sparsity) const {
  float min = 0, max = 0;
  FindMinMax(values, &min, &max, len);

  if (MIN_MAX_QUANTIZATION == kind) {
    return ChooseQuantizationParams(min, max, precision, preserve_sparsity);
  }
  else {
    if (0 == len) {
      return ChooseQuantizationParams(min, max, precision, preserve_sparsity);
    }

    Histogram hist(2048, min, max);
    for (int i = 0; i < len; ++i) {
      hist.Add(values[i]);
    }

    return ChooseQuantizationParams(hist, kind, precision, preserve_sparsity);
  }
}

TensorQuantizationParams QuantizationFactory::ChooseQuantizationParams(
    const float *values, int len, bool is_weight) const {
  if (is_weight) {
    return ChooseQuantizationParams(
        values, len, GetWeightKind(), GetWeightPrecision(),
        GetPreserveWeightSparsity());
  }
  else {
    return ChooseQuantizationParams(
        values, len, GetActivationKind(), GetActivationPrecision(),
        GetPreserveActivationSparsity());
  }
}

void QuantizationFactory::ChooseRequantizationMultiplier_(
    float real_multiplier,
    int32_t* quantized_multiplier,
    int* right_shift) const {

  assert(real_multiplier != 0.f);

  // Assuming requantization_multiplier_precision_ = 31,
  // the default right shift is 31 when the real multiplier is already
  // in interval [1/2, 1).
  // Multiplying a 32-bit signed integer with all 31 bits except the sign bit
  // is used followed by 31-bit right shift implements multiplying with a real
  // number in [1/2, 1).
  // We want to utilize all 31 bits except the sign bit in the 32-bit signed
  // integer to get the best accuracy.
  int s = 31;

  // We want to bring the real multiplier into the interval [1/2, 1).
  // We can do so by multiplying it by two, and recording how many times
  // we multiplied by two so that we can compensate that by a right
  // shift by the same amount.
  if (real_multiplier > 0.f) {
    while (real_multiplier < 0.5f) {
      real_multiplier *= 2.f;
      s++;
    }
    while (real_multiplier > 1.f) {
      real_multiplier /= 2.f;
      s--;
    }
  }
  // Now that the real multiplier is in [1/2, 1), we convert it
  // into a fixed-point number.
  int64_t q = static_cast<int64_t>(
      round(
        real_multiplier * (1ll << (requantization_multiplier_precision_ - 1))));
  assert(q <= (1ll << (requantization_multiplier_precision_ - 1)));
  // Handle the special case when the real multiplier was so close to 1
  // that its fixed-point approximation was undistinguishable from 1.
  // We handle this by dividing it by two, and remembering to decrement
  // the right shift amount.
  if (q == (1ll << (requantization_multiplier_precision_ - 1))) {
    q /= 2;
    s--;
  }
  assert(s >= 0);
  assert(q >= 0);
  assert(q <= numeric_limits<int32_t>::max());
  *quantized_multiplier = static_cast<int32_t>(q);
  *right_shift = s;
  assert(s < 64);
}

RequantizationParams QuantizationFactory::ChooseRequantizationMultiplier(
    float real_multiplier,
    TensorQuantizationParams target_qparams) const {
  RequantizationParams params;
  params.target_qparams = target_qparams;
  params.real_multiplier = real_multiplier;

  ChooseRequantizationMultiplier_(
    real_multiplier, &params.multiplier, &params.right_shift);

  return params;
}

void FindMinMax(const float *a, float* min, float* max, int len) {
  if (len <= 0) {
    *min = 0.0f;
    *max = 0.0f;
    return;
  }

  float temp_min = *a, temp_max = *a;
  int i = 0;

#ifdef __AVX__
  __m256 min_v = _mm256_set1_ps(*a), max_v = _mm256_set1_ps(*a);
  constexpr int VLEN = 8;
  if (len >= VLEN) {
    for ( ; i < len / VLEN * VLEN; i += VLEN) {
      min_v = _mm256_min_ps(min_v, _mm256_loadu_ps(a + i));
      max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(a + i));
    }

    float min_buf[VLEN], max_buf[VLEN];
    _mm256_storeu_ps(min_buf, min_v);
    _mm256_storeu_ps(max_buf, max_v);
    for (int j = 0; j < VLEN; ++j) {
      temp_min = std::min(temp_min, min_buf[j]);
      temp_max = std::max(temp_max, max_buf[j]);
    }
  }
#endif

  for ( ; i < len; i++) {
    temp_min = std::min(temp_min, a[i]);
    temp_max = std::max(temp_max, a[i]);
  }
  *min = temp_min;
  *max = temp_max;
}

} // namespace dnnlowp
