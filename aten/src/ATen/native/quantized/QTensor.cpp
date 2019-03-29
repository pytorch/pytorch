#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/quantized/QTensorImpl.h>


namespace at {
namespace native {

QTensor quantize_linear(const RealTensor& self, double scale, int64_t zero_point) {
  auto quantizer = make_per_tensor_affine_quantizer(scale, zero_point);
  return quantizer->quantize(self);
}

RealTensor dequantize(const QTensor& self) {
  return get_qtensorimpl(self)->quantizer()->dequantize(self);
}

Scalar q_scale(const QTensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  AT_ASSERT(quantizer->qscheme() == kPerTensorAffine);
  return Scalar(static_cast<PerTensorAffineQuantizer*>(quantizer.get())->scale());
}

Scalar q_zero_point(const QTensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  AT_ASSERT(quantizer->qscheme() == kPerTensorAffine);
  return Scalar(static_cast<PerTensorAffineQuantizer*>(quantizer.get())->zero_point());
}

Quantizer* quantizer(const QTensor& self) {
  return get_qtensorimpl(self)->quantizer().get();
}

} // namespace native
} // namespace at
