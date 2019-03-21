#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Quantizer.h>
#include <ATen/QTensorImpl.h>


namespace at {
namespace native {

// This is an internal utility function for getting at the QTensorImpl,
// You should only use this for writing low level
// setters/getters for QTensorImpl fields; otherwise, you should use
// the low level setters/getters that were implemented using this.
// This may be called repeatedly, so make sure it's pretty cheap.
inline QTensorImpl* get_qtensorimpl(const QTensor& self) {
  // TODO: remove this when Variable and Tensor are merged
  AT_ASSERTM(!self.is_variable(), "_internal_get_QTensorImpl: should not be a variable");
  // TODO: uncomment
  // AT_ASSERTM(self.is_quantized(), "_internal_get_QTensorImpl: not a quantized tensor");
  return static_cast<QTensorImpl*>(self.unsafeGetTensorImpl());
}


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
