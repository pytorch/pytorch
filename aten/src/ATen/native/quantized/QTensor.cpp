#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Quantizer.h>
#include <ATen/QTensorImpl.h>


namespace at {
namespace native {

QTensor quantize_linear(const RealTensor& self, double scale, int64_t zero_point) {
  auto quantizer = make_per_layer_affine_quantizer(scale, zero_point);
  return quantizer->quantize(self);
}

RealTensor dequantize(const QTensor& self) {
  return get_qtensorimpl(self)->quantizer()->dequantize(self);
}

Scalar q_scale(const QTensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  // TODO: qscheme?
  AT_ASSERT(quantizer->name() == "PerLayerAffineQuantizer");
  return Scalar(static_cast<PerLayerAffineQuantizer*>(quantizer)->scale());
}

Scalar q_zero_point(const QTensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  // TODO: qscheme?
  AT_ASSERT(quantizer->name() == "PerLayerAffineQuantizer");
  return Scalar(static_cast<PerLayerAffineQuantizer*>(quantizer)->zero_point());
}

Quantizer* quantizer(const QTensor& self) {
  return get_qtensorimpl(self)->quantizer();
}

} // namespace native
} // namespace at
