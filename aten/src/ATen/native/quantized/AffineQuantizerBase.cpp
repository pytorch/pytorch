#include <ATen/native/quantized/AffineQuantizerBase.h>
#include <c10/util/irange.h>
#include <climits>

#ifdef USE_FBGEMM
#include <fbgemm/QuantUtils.h>
#endif
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif


namespace at::native {

namespace {

template <typename T>
void checkZeroPoint(const std::string& fn_name, int64_t zero_point) {
  TORCH_CHECK(
      zero_point <= std::numeric_limits<T>::max(),
      fn_name,
      " zero_point ",
      zero_point,
      " is out of range.");
  TORCH_CHECK(
      zero_point >= std::numeric_limits<T>::min(),
      fn_name,
      " zero_point ",
      zero_point,
      " is out of range.");
}

} // anonymous namespace

#ifdef USE_FBGEMM
// Note: quantize_val is only explicitly used in test outside of this file
template <typename T>
T quantize_val(double scale, int64_t zero_point, float value) {
  // Internally, fbgemm::Quantize uses std::nearbyint.
  // std::nearbyint results in nearest integer value according to the current
  // rounding mode and the default rounding mode is rounds to even in half-way
  // cases in most popular processor architectures like x86 and ARM. This is
  // typically faster than an alternatives like std::round that rounds half-way
  // cases away from zero, and can be consistent with SIMD implementations for
  // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
  // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto qvalue = fbgemm::Quantize<typename T::underlying, false /*LEGACY*/>(
      value,
      static_cast<int32_t>(zero_point),
      static_cast<float>(scale),
      /*result_precision=*/CHAR_BIT * sizeof(typename T::underlying));
  return static_cast<T>(qvalue);
}

template <typename T, int precision>
void quantize_vec(
    double scale,
    int64_t zero_point,
    const float* src,
    T* dst,
    size_t count) {
  fbgemm::Quantize<typename T::underlying, false /*LEGACY*/>(
      src,
      (typename T::underlying*)dst,
      count,
      fbgemm::TensorQuantizationParams{
          static_cast<float>(scale), static_cast<int32_t>(zero_point), precision});
}

#if defined(__ARM_NEON__) || defined(__aarch64__)
// For use when compiling FBGEMM on aarch64 but still supporting x86
// intrinsics via simde
template <typename T>
T quantize_val_arm(
    const float scale,
    const int32_t zero_point,
    const float value) {
  constexpr int32_t qmin = std::numeric_limits<T>::min();
  constexpr int32_t qmax = std::numeric_limits<T>::max();
  float inv_scale = 1.0f / scale;
  auto r = zero_point + static_cast<int32_t>(std::nearbyint(value * inv_scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<T>(r);
}

template uint8_t quantize_val_arm<uint8_t>(
    const float scale,
    const int32_t zero_point,
    const float value);
template int8_t quantize_val_arm<int8_t>(
    const float scale,
    const int32_t zero_point,
    const float value);
#endif

template <typename T>
inline float dequantize_val(double scale, int64_t zero_point, T value) {
  fbgemm::TensorQuantizationParams qparams{};
  qparams.scale = static_cast<float>(scale);
  qparams.zero_point = static_cast<int32_t>(zero_point);
  return fbgemm::Dequantize<typename T::underlying>(value.val_, qparams);
}
#else // USE_FBGEMM

template <typename T>
T quantize_val(double scale, int64_t zero_point, float value) {
  // std::nearbyint results in nearest integer value according to the current
  // rounding mode and the default rounding mode is rounds to even in half-way
  // cases in most popular processor architectures like x86 and ARM. This is
  // typically faster than an alternatives like std::round that rounds half-way
  // cases away from zero, and can be consistent with SIMD implementations for
  // example in x86 using _mm512_cvtps_epi32 or mm512_round_ps with
  // _MM_FROUND_CUR_DIRECTION option that also follow the current rounding mode.
  int64_t qvalue;
  constexpr int64_t qmin = std::numeric_limits<typename T::underlying>::min();
  constexpr int64_t qmax = std::numeric_limits<typename T::underlying>::max();
  float inv_scale = 1.0f / static_cast<float>(scale);
  qvalue = static_cast<int64_t>(zero_point + std::nearbyint(value * inv_scale));
  qvalue = std::max<int64_t>(qvalue, qmin);
  qvalue = std::min<int64_t>(qvalue, qmax);
  return static_cast<T>(qvalue);
}

template <typename T>
T quantize_val_arm(
    const float scale,
    const int32_t zero_point,
    const float value) {
  constexpr int32_t qmin = std::numeric_limits<T>::min();
  constexpr int32_t qmax = std::numeric_limits<T>::max();
  float inv_scale = 1.0f / scale;
#ifndef _MSC_VER
  auto r = static_cast<int32_t>(std::nearbyint(value * inv_scale));
  // builtin_add_overflow() returns true in case of overflow
  if (__builtin_add_overflow(zero_point, r, &r)) {
    // zero_point must be a non-negative value between qmin and qmax,
    // i.e. only overflow can happen.
    r = qmax;
  }
#else
  auto r = zero_point + static_cast<int32_t>(std::nearbyint(value * inv_scale));
#endif
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<T>(r);
}

template <typename T, int precision>
void quantize_vec(
    double scale,
    int64_t zero_point,
    const float* src,
    T* dst,
    size_t count) {
  checkZeroPoint<typename T::underlying>("quantize_vec", zero_point);
  for (const auto i : c10::irange(count)) {
    dst[i] = quantize_val<T>(scale, zero_point, src[i]);
  }
}

template uint8_t quantize_val_arm<uint8_t>(
    const float scale,
    const int32_t zero_point,
    const float value);
template int8_t quantize_val_arm<int8_t>(
    const float scale,
    const int32_t zero_point,
    const float value);
template <typename T>
TORCH_API float dequantize_val(double scale, int64_t zero_point, T value) {
  return static_cast<float>(scale) * (value.val_ - static_cast<int32_t>(zero_point));
}
#endif // USE_FBGEMM

/*
* Quantize value based on the following equation
* Xq = std::nearbyint(Xf * inv_scale + zero_point)
* where zero_point is in float.
*
* Note: For the case of embedding quantization we will set zero_point
* to (-Xmin/scale), where Xmin is the min value in input tensor row.
*/
int quantize_val_float_qparams(float scale, float zero_point, float value, int qmin, int qmax) {

  float inv_scale = scale == 0 ? 1.0f : 1.0f / scale;
  auto qvalue = static_cast<int>(lrintf(value * inv_scale + zero_point));
  qvalue = std::max(qmin, std::min(qvalue, qmax));
  return qvalue;
}

template <typename SRC_T, typename DST_T>
DST_T requantize_val(
    double src_scale,
    int64_t src_zero_point,
    double dst_scale,
    int64_t dst_zero_point,
    SRC_T src) {
  const auto dq = dequantize_val<SRC_T>(src_scale, src_zero_point, src);
  return quantize_val<DST_T>(dst_scale, dst_zero_point, dq);
}

template <typename DST_T>
DST_T requantize_from_int(double multiplier, int64_t zero_point, int64_t src) {
  int64_t quantize_down =
      zero_point + lrintf(static_cast<float>(static_cast<double>(src) * multiplier));
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  int32_t min = std::numeric_limits<typename DST_T::underlying>::min();
  int32_t max = std::numeric_limits<typename DST_T::underlying>::max();
  return static_cast<DST_T>(
      std::min<int64_t>(std::max<int64_t>(quantize_down, min), max));
}

template TORCH_API qint8
quantize_val<qint8>(double scale, int64_t zero_point, float value);
template TORCH_API quint8
quantize_val<quint8>(double scale, int64_t zero_point, float value);
template TORCH_API qint32
quantize_val<qint32>(double scale, int64_t zero_point, float value);
template TORCH_API void quantize_vec<c10::qint8>(
    double scale,
    int64_t zero_point,
    const float* src,
    c10::qint8* dst,
    size_t count);
template TORCH_API void quantize_vec<c10::quint8>(
    double scale,
    int64_t zero_point,
    const float* src,
    c10::quint8* dst,
    size_t count);
template TORCH_API void quantize_vec<c10::qint32, 32>(
    double scale,
    int64_t zero_point,
    const float* src,
    c10::qint32* dst,
    size_t count);

template TORCH_API float dequantize_val<qint8>(
    double scale,
    int64_t zero_point,
    qint8 value);
template TORCH_API float dequantize_val<quint8>(
    double scale,
    int64_t zero_point,
    quint8 value);
template TORCH_API float dequantize_val<qint32>(
    double scale,
    int64_t zero_point,
    qint32 value);

template TORCH_API qint8
requantize_val<qint8, qint8>(double, int64_t, double, int64_t, qint8);
template TORCH_API quint8
requantize_val<qint8, quint8>(double, int64_t, double, int64_t, qint8);
template TORCH_API qint32
requantize_val<qint8, qint32>(double, int64_t, double, int64_t, qint8);
template TORCH_API qint8
requantize_val<quint8, qint8>(double, int64_t, double, int64_t, quint8);
template TORCH_API quint8
requantize_val<quint8, quint8>(double, int64_t, double, int64_t, quint8);
template TORCH_API qint32
requantize_val<quint8, qint32>(double, int64_t, double, int64_t, quint8);
template TORCH_API qint8
requantize_val<qint32, qint8>(double, int64_t, double, int64_t, qint32);
template TORCH_API quint8
requantize_val<qint32, quint8>(double, int64_t, double, int64_t, qint32);
template TORCH_API qint32
requantize_val<qint32, qint32>(double, int64_t, double, int64_t, qint32);

template TORCH_API qint8 requantize_from_int<qint8>(double, int64_t, int64_t);
template TORCH_API quint8
requantize_from_int<quint8>(double, int64_t, int64_t);
template TORCH_API qint32
requantize_from_int<qint32>(double, int64_t, int64_t);

} // namespace at::native
