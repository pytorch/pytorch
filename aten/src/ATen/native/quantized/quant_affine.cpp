#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/quantized/quant_affine.h>

namespace at {
namespace native {

template <typename T>
void checkZeroPoint(std::string fn_name, int64_t zero_point) {
  TORCH_CHECK(zero_point <= std::numeric_limits<T>::max(),
              fn_name,
              " zero_point ",
              zero_point,
              " is out of range.");
  TORCH_CHECK(zero_point >= std::numeric_limits<T>::min(),
              fn_name,
              " zero_point ",
              zero_point,
              " is out of range.");
}

DEFINE_DISPATCH(quantize_tensor_affine_stub);
DEFINE_DISPATCH(quantize_tensor_per_channel_affine_stub);
DEFINE_DISPATCH(dequantize_tensor_affine_stub);
DEFINE_DISPATCH(dequantize_tensor_per_channel_affine_stub);

namespace {

void checkCPUTensor(std::string fn_name, Tensor t) {
  TORCH_CHECK(
      t.device() == kCPU,
      fn_name,
      " expects a CPU Tensor.");
}

void checkFloatTensor(std::string fn_name, Tensor t) {
  TORCH_CHECK(
      t.scalar_type() == kFloat,
      fn_name,
      " expects a Float Tensor.");
}

void checkSameDevice(std::string fn_name, Tensor t1, Tensor t2) {
  TORCH_CHECK(
      t1.device() == t2.device(),
      fn_name,
      " expects a quantized and float tensors to be on the same device.");
}

template <typename T>
void checkQuantizedTensor(std::string fn_name, Tensor t) {
  TORCH_CHECK(t.is_quantized(),
           fn_name,
           " expects a quantized Tensor.");
  TORCH_CHECK(t.scalar_type() == caffe2::TypeMeta::Make<T>(),
           fn_name,
           " expects a ",
           caffe2::TypeMeta::Make<T>(),
           " Tensor");
  TORCH_CHECK(t.is_cuda() || t.device() == kCPU,
           fn_name,
           " expects a CUDA or CPU quantized Tensor");
}

template <typename T>
void checkZeroPoints(std::string fn_name, Tensor zero_points) {
  auto zero_points_data = zero_points.data_ptr<int64_t>();
  for (size_t i = 0; i < zero_points.numel(); ++i) {
    native::checkZeroPoint<T>(fn_name, zero_points_data[i]);
  }
}

void checkSameSize(std::string fn_name, Tensor qt, Tensor rt) {
  TORCH_CHECK(
      qt.sizes().equals(rt.sizes()),
      fn_name, 
      " only works with Tensors with the same shape");
}

} // anonymous namespace

Tensor quantize_tensor_affine(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point) {
  auto fn_name = "quantize_tensor_affine";
  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {  
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    native::checkZeroPoint<underlying_t>(fn_name, zero_point);
  });

  quantize_tensor_affine_stub(rtensor.device().type(), rtensor, qtensor, scale, zero_point);
  return qtensor;
}

Tensor quantize_tensor_per_channel_affine(Tensor rtensor,
                                                     Tensor qtensor,
                                                     Tensor scales,
                                                     Tensor zero_points,
                                                     int64_t axis) {
  auto fn_name = "quantize_tensor_per_channel_affine";

  checkFloatTensor(fn_name, rtensor);
  checkCPUTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {  
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    native::checkZeroPoints<underlying_t>(fn_name, zero_points);
  });

  TORCH_CHECK(0 <= axis && axis < rtensor.dim(), "Channel axis out of range in per channel affine quantization.");
  int64_t channel = rtensor.size(axis);
  TORCH_CHECK(channel == int64_t(scales.numel()), "length of scales must equal to channel");
  TORCH_CHECK(channel == int64_t(zero_points.numel()), "length of zero_points must equal to channel");

  quantize_tensor_per_channel_affine_stub(rtensor.device().type(), rtensor, qtensor, scales, zero_points, axis);
  return qtensor;
}

Tensor dequantize_tensor_affine(Tensor qtensor, Tensor rtensor, double scale, int64_t zero_point) {
  auto fn_name = "dequantize_tensor_affine";
  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {  
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    native::checkZeroPoint<underlying_t>(fn_name, zero_point);
  });

  dequantize_tensor_affine_stub(qtensor.device().type(), qtensor, rtensor, scale, zero_point);
  return rtensor;
}

Tensor dequantize_tensor_per_channel_affine(Tensor qtensor,
                                            Tensor rtensor,
                                            Tensor scales,
                                            Tensor zero_points,
                                            int64_t axis) {
  auto fn_name = "dequantize_tensor_per_channel_affine";

  checkFloatTensor(fn_name, rtensor);
  checkCPUTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {  
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    native::checkZeroPoints<underlying_t>(fn_name, zero_points);
  });

  TORCH_CHECK(0 <= axis && axis < qtensor.dim(), "Channel axis out of range in per channel affine dequantization.");
  int64_t channel = qtensor.size(axis);
  TORCH_CHECK(channel == int64_t(scales.numel()), "length of scales must equal to channel");
  TORCH_CHECK(channel == int64_t(zero_points.numel()), "length of zero_points must equal to channel");

  dequantize_tensor_per_channel_affine_stub(qtensor.device().type(), qtensor, rtensor, scales, zero_points, axis);
  return rtensor;

}

} // native
} // at
