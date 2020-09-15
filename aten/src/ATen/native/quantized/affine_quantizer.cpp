#include <ATen/native/quantized/affine_quantizer.h>
#include <cfenv>

#ifdef USE_FBGEMM
#include <fbgemm/QuantUtils.h>
#endif
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace at {
namespace native {

DEFINE_DISPATCH(quantize_tensor_per_tensor_affine_stub);
DEFINE_DISPATCH(quantize_tensor_per_channel_affine_stub);
DEFINE_DISPATCH(quantize_tensor_per_channel_float_qparams_stub);
DEFINE_DISPATCH(dequantize_tensor_per_tensor_affine_stub);
DEFINE_DISPATCH(dequantize_tensor_per_channel_affine_stub);
DEFINE_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub);
DEFINE_DISPATCH(quantize_tensor_per_tensor_affine_sub_byte_stub);
DEFINE_DISPATCH(dequantize_tensor_per_tensor_affine_sub_byte_stub);

namespace {

void checkRoundingMode(const std::string& fn_name) {
// Disabling this warning message for now as it is printed incorrectly. Need to fix

/*  TORCH_WARN_ONCE(
      std::fegetround() != FE_TONEAREST,
      fn_name,
      " current rounding mode is not set to round-to-nearest-ties-to-even (FE_TONEAREST). This will cause accuracy issues in quantized models.");
*/
  return;
}

void checkCPUTensor(const std::string& fn_name, Tensor t) {
  TORCH_CHECK(
      t.device().type() == kCPU, fn_name, " only supports CPU device type.");
}

void checkFloatTensor(const std::string& fn_name, Tensor t) {
  TORCH_CHECK(t.scalar_type() == kFloat, fn_name, " expects a Float Tensor.");
}

void checkSameDevice(const std::string& fn_name, Tensor t1, Tensor t2) {
  TORCH_CHECK(
      t1.device() == t2.device(),
      fn_name,
      " expects a quantized and float tensors to be on the same device.");
}

template <typename T>
void checkQuantizedTensor(const std::string& fn_name, Tensor t) {
  TORCH_CHECK(t.is_quantized(), fn_name, " expects a quantized Tensor.");
  TORCH_CHECK(
      t.scalar_type() == caffe2::TypeMeta::Make<T>(),
      fn_name,
      " expects a ",
      caffe2::TypeMeta::Make<T>(),
      " Tensor, got ",
      t.scalar_type());
}

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

template <typename T>
void checkZeroPoints(const std::string& fn_name, Tensor zero_points) {
  auto zero_points_data = zero_points.data_ptr<int64_t>();
  for (size_t i = 0; i < zero_points.numel(); ++i) {
    checkZeroPoint<T>(fn_name, zero_points_data[i]);
  }
}

void checkSameSize(const std::string& fn_name, Tensor qt, Tensor rt) {
  TORCH_CHECK(
      qt.sizes().equals(rt.sizes()),
      fn_name,
      " only works with Tensors with the same shape");
}

} // anonymous namespace

Tensor quantize_tensor_per_tensor_affine(
    Tensor rtensor,
    Tensor qtensor,
    double scale,
    int64_t zero_point) {
  static const auto fn_name = "quantize_tensor_per_tensor_affine";

  checkRoundingMode(fn_name);
  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    checkZeroPoint<underlying_t>(fn_name, zero_point);
  });

  // Temporary solution to pack the tensor if dtype is torch.quint4x2
  // Can move this into the fbgemm::Quantize op.
  if (qtensor.scalar_type() == at::ScalarType::QUInt4x2) {
    quantize_tensor_per_tensor_affine_sub_byte_stub(
      rtensor.device().type(), rtensor, qtensor, scale, zero_point);
  }
  else {
    quantize_tensor_per_tensor_affine_stub(
      rtensor.device().type(), rtensor, qtensor, scale, zero_point);
  }
  return qtensor;
}

Tensor quantize_tensor_per_channel_affine(
    Tensor rtensor,
    Tensor qtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis) {
  static const auto fn_name = "quantize_tensor_per_channel_affine";

  checkRoundingMode(fn_name);
  checkFloatTensor(fn_name, rtensor);
  checkCPUTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    checkZeroPoints<underlying_t>(fn_name, zero_points);
  });

  TORCH_CHECK(
      0 <= axis && axis < rtensor.dim(),
      "Channel axis out of range in per channel affine quantization. Got: ",
      axis, "Expected: [0, ", rtensor.dim(), ")");
  int64_t channel = rtensor.size(axis);
  TORCH_CHECK(
      channel == int64_t(scales.numel()),
      "length of scales must equal to channel");
  TORCH_CHECK(
      channel == int64_t(zero_points.numel()),
      "length of zero_points must equal to channel");

  quantize_tensor_per_channel_affine_stub(
      rtensor.device().type(), rtensor, qtensor, scales, zero_points, axis);
  return qtensor;
}

Tensor quantize_tensor_per_channel_float_qparams(
    Tensor rtensor,
    Tensor qtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis) {
  static const auto fn_name = "quantize_tensor_per_channel_float_qparams";

  checkRoundingMode(fn_name);
  checkFloatTensor(fn_name, rtensor);
  checkCPUTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
  });

  TORCH_CHECK(
      0 <= axis && axis < rtensor.dim(),
      "Channel axis out of range in per channel float qparams quantization. Got: ",
      axis, "Expected: [0, ", rtensor.dim(), ")");
  int64_t channel = rtensor.size(axis);
  TORCH_CHECK(
      channel == int64_t(scales.numel()),
      "length of scales must equal to channel");
  TORCH_CHECK(
      channel == int64_t(zero_points.numel()),
      "length of zero_points must equal to channel");

  quantize_tensor_per_channel_float_qparams_stub(
      rtensor.device().type(), rtensor, qtensor, scales, zero_points, axis);
  return qtensor;

}

Tensor dequantize_tensor_per_tensor_affine(
    Tensor qtensor,
    Tensor rtensor,
    double scale,
    int64_t zero_point) {
  static const auto fn_name = "dequantize_tensor_per_tensor_affine";
  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    checkZeroPoint<underlying_t>(fn_name, zero_point);
  });

  if (qtensor.scalar_type() == at::ScalarType::QUInt4x2) {
    dequantize_tensor_per_tensor_affine_sub_byte_stub(
        qtensor.device().type(), qtensor, rtensor, scale, zero_point);
  } else {
    dequantize_tensor_per_tensor_affine_stub(
        qtensor.device().type(), qtensor, rtensor, scale, zero_point);
  }
  return rtensor;
}

Tensor dequantize_tensor_per_channel_affine(
    Tensor qtensor,
    Tensor rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis) {
  static const auto fn_name = "dequantize_tensor_per_channel_affine";

  checkFloatTensor(fn_name, rtensor);
  checkCPUTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    checkZeroPoints<underlying_t>(fn_name, zero_points);
  });

  TORCH_CHECK(
      0 <= axis && axis < qtensor.dim(),
      "Channel axis out of range in per channel affine dequantization. Got:",
      axis, " Expected: [0, ", qtensor.dim(), ")");
  int64_t channel = qtensor.size(axis);
  TORCH_CHECK(
      channel == int64_t(scales.numel()),
      "length of scales must equal to channel");
  TORCH_CHECK(
      channel == int64_t(zero_points.numel()),
      "length of zero_points must equal to channel");

  dequantize_tensor_per_channel_affine_stub(
      qtensor.device().type(), qtensor, rtensor, scales, zero_points, axis);
  return rtensor;
}

Tensor dequantize_tensor_per_channel_float_qparams(
    Tensor qtensor,
    Tensor rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis) {
  static const auto fn_name = "dequantize_tensor_per_channel_affine";

  checkFloatTensor(fn_name, rtensor);
  checkCPUTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
  });

  TORCH_CHECK(
      0 <= axis && axis < qtensor.dim(),
      "Channel axis out of range in per channel float qparams dequantization. Got:",
      axis, " Expected: [0, ", qtensor.dim(), ")");
  int64_t channel = qtensor.size(axis);
  TORCH_CHECK(
      channel == int64_t(scales.numel()),
      "length of scales must equal to channel");
  TORCH_CHECK(
      channel == int64_t(zero_points.numel()),
      "length of zero_points must equal to channel");

  dequantize_tensor_per_channel_float_qparams_stub(
      qtensor.device().type(), qtensor, rtensor, scales, zero_points, axis);
  return rtensor;
}

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
  int32_t qvalue;
  qvalue = fbgemm::Quantize<typename T::underlying, false /*LEGACY*/>(
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
          (float)scale, (int32_t)zero_point, precision});
}

template <typename T>
inline float dequantize_val(double scale, int64_t zero_point, T value) {
  fbgemm::TensorQuantizationParams qparams;
  qparams.scale = static_cast<float>(scale);
  qparams.zero_point = static_cast<int32_t>(zero_point);
  return fbgemm::Dequantize<typename T::underlying>(value.val_, qparams);
}
#else // USE_FBGEMM

#if defined(__ANDROID__) && !defined(__NDK_MAJOR__)
template <class T>
inline float Round(const float x) {
  return ::nearbyintf(x);
}
inline double Round(const double x) {
  return ::nearbyint(x);
}
#else
template <class T>
inline T Round(const T x) {
  return std::nearbyint(x);
}
#endif

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
  qvalue = static_cast<int64_t>(zero_point + Round(value * inv_scale));
  qvalue = std::max<int64_t>(qvalue, qmin);
  qvalue = std::min<int64_t>(qvalue, qmax);
  return static_cast<T>(qvalue);
}

uint8_t quantize_val_arm(
    const float scale,
    const int32_t zero_point,
    const float value) {
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();
  auto r = zero_point + static_cast<int32_t>(Round(value / scale));
  r = std::max(r, qmin);
  r = std::min(r, qmax);
  return static_cast<uint8_t>(r);
}

template <typename T, int precision>
void quantize_vec(
    double scale,
    int64_t zero_point,
    const float* src,
    T* dst,
    size_t count) {
  checkZeroPoint<typename T::underlying>("quantize_vec", zero_point);
  for (int64_t i = 0; i < count; ++i) {
    dst[i] = quantize_val<T>(scale, zero_point, src[i]);
  }
}

template <typename T>
CAFFE2_API float dequantize_val(double scale, int64_t zero_point, T value) {
  // We need to convert the qint8 value to float to ensure the subtraction
  // subexpression returns a float
  return (static_cast<float>(value.val_) - zero_point) * scale;
}
#endif // USE_FBGEMM

/*
* Quantize value based on the following equation
* Xq = Round(Xf * inv_scale + zero_point)
* where zero_point is in float.
*
* Note: For the case of embedding quantization we will set zero_point
* to (-Xmin/scale), where Xmin is the min value in input tensor row.
*/
template <typename T>
T quantize_val_float_qparams(float scale, float zero_point, float value) {
  int64_t qvalue;

  // TODO make sure qmax and qmin for dtypes other than int8, uint8 is correctly defined.
  constexpr int64_t qmin = std::numeric_limits<typename T::underlying>::min();
  constexpr int64_t qmax = std::numeric_limits<typename T::underlying>::max();
  float inv_scale = scale == 0 ? 1.0f : 1.0f / scale;
  qvalue = lrintf(value * inv_scale + zero_point);
  qvalue = std::max(qmin, std::min(qvalue, qmax));
  return static_cast<T>(qvalue);
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
      zero_point + lrintf(src * static_cast<float>(multiplier));
  int32_t min = std::numeric_limits<typename DST_T::underlying>::min();
  int32_t max = std::numeric_limits<typename DST_T::underlying>::max();
  return static_cast<DST_T>(
      std::min<int64_t>(std::max<int64_t>(quantize_down, min), max));
}

template CAFFE2_API qint8
quantize_val<qint8>(double scale, int64_t zero_point, float value);
template CAFFE2_API quint8
quantize_val<quint8>(double scale, int64_t zero_point, float value);
template CAFFE2_API qint32
quantize_val<qint32>(double scale, int64_t zero_point, float value);
template CAFFE2_API void quantize_vec<c10::qint8>(
    double scale,
    int64_t zero_point,
    const float* src,
    c10::qint8* dst,
    size_t count);
template CAFFE2_API void quantize_vec<c10::quint8>(
    double scale,
    int64_t zero_point,
    const float* src,
    c10::quint8* dst,
    size_t count);
template CAFFE2_API void quantize_vec<c10::qint32, 32>(
    double scale,
    int64_t zero_point,
    const float* src,
    c10::qint32* dst,
    size_t count);

template CAFFE2_API float dequantize_val<qint8>(
    double scale,
    int64_t zero_point,
    qint8 value);
template CAFFE2_API float dequantize_val<quint8>(
    double scale,
    int64_t zero_point,
    quint8 value);
template CAFFE2_API float dequantize_val<qint32>(
    double scale,
    int64_t zero_point,
    qint32 value);

template CAFFE2_API qint8
requantize_val<qint8, qint8>(double, int64_t, double, int64_t, qint8);
template CAFFE2_API quint8
requantize_val<qint8, quint8>(double, int64_t, double, int64_t, qint8);
template CAFFE2_API qint32
requantize_val<qint8, qint32>(double, int64_t, double, int64_t, qint8);
template CAFFE2_API qint8
requantize_val<quint8, qint8>(double, int64_t, double, int64_t, quint8);
template CAFFE2_API quint8
requantize_val<quint8, quint8>(double, int64_t, double, int64_t, quint8);
template CAFFE2_API qint32
requantize_val<quint8, qint32>(double, int64_t, double, int64_t, quint8);
template CAFFE2_API qint8
requantize_val<qint32, qint8>(double, int64_t, double, int64_t, qint32);
template CAFFE2_API quint8
requantize_val<qint32, quint8>(double, int64_t, double, int64_t, qint32);
template CAFFE2_API qint32
requantize_val<qint32, qint32>(double, int64_t, double, int64_t, qint32);

template CAFFE2_API qint8 requantize_from_int<qint8>(double, int64_t, int64_t);
template CAFFE2_API quint8
requantize_from_int<quint8>(double, int64_t, int64_t);
template CAFFE2_API qint32
requantize_from_int<qint32>(double, int64_t, int64_t);

template CAFFE2_API qint8
quantize_val_float_qparams<qint8>(float scale, float zero_point, float value);
template CAFFE2_API quint8
quantize_val_float_qparams<quint8>(float scale, float zero_point, float value);
template CAFFE2_API qint32
quantize_val_float_qparams<qint32>(float scale, float zero_point, float value);
} // namespace native
} // namespace at
