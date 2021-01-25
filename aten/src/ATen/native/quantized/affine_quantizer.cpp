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

  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
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

  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
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

} // namespace native
} // namespace at
