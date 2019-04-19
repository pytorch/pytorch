#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/quantized/QTensorImpl.h>


namespace at {
namespace native {

QTensor quantize_linear_cpu(const RealTensor& self, double scale, int64_t zero_point) {
  auto quantizer = make_per_tensor_affine_quantizer(scale, zero_point);
  return quantizer->quantize(self);
}

RealTensor dequantize_quant(const QTensor& self) {
  return get_qtensorimpl(self)->quantizer()->dequantize(self);
}

Scalar q_scale_quant(const QTensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  AT_ASSERT(quantizer->qscheme() == kPerTensorAffine);
  return Scalar(static_cast<PerTensorAffineQuantizer*>(quantizer.get())->scale());
}

Scalar q_zero_point_quant(const QTensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  AT_ASSERT(quantizer->qscheme() == kPerTensorAffine);
  return Scalar(static_cast<PerTensorAffineQuantizer*>(quantizer.get())->zero_point());
}

Quantizer* quantizer(const QTensor& self) {
  return get_qtensorimpl(self)->quantizer().get();
}

Tensor int_repr_quant(const QTensor& self) {
  Tensor dst = at::empty(self.sizes(), self.options().dtype(at::kByte));
  uint8_t* self_data = reinterpret_cast<uint8_t *>(self.data<qint8>());
  uint8_t* dst_data = dst.data<uint8_t>();
  memcpy(dst_data, self_data, self.numel());
  return dst;
}

} // namespace native
} // namespace at
