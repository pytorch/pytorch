#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/quantized/QTensorImpl.h>


namespace at {
namespace native {

Tensor quantize_linear_cpu(const Tensor& self, double scale, int64_t zero_point, optional<ScalarType> dtype) {
  auto quantizer = make_per_tensor_affine_quantizer(scale, zero_point, dtype);
  return quantizer->quantize(self);
}

Tensor quantize_linear_cpu(const Tensor& self, double scale, int64_t zero_point, ScalarType dtype) {
  return quantize_linear_cpu(self, scale, zero_point, optional<ScalarType>(dtype));
}

Tensor quantize_linear_cpu(const Tensor& self, double scale, int64_t zero_point) {
  return quantize_linear_cpu(self, scale, zero_point, c10::nullopt);
}

Tensor dequantize_quant(const Tensor& self) {
  return get_qtensorimpl(self)->quantizer()->dequantize(self);
}

Tensor dequantize_linear_cpu(const Tensor& self, double scale, int64_t zero_point, ScalarType dtype) {
  AT_CHECK(isUnderlying(typeMetaToScalarType(self.dtype()), dtype), "Scalar type for quantized Tensor must have same underlying type as input");
  Tensor t = at::_empty_affine_quantized(self.sizes(), self.options().dtype(dtype), scale, zero_point);
  AT_DISPATCH_QINT_TYPES(t.scalar_type(), "dequantize_linear_cpu", [&]() {
     scalar_t* qdata = t.data<scalar_t>();
     auto* self_data = self.data<underlying_t>();
     for (int i = 0; i < t.numel(); ++i) {
       qdata[i] = static_cast<scalar_t>(self_data[i]);
     }});
  return t;
}

Scalar q_scale_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  AT_ASSERT(quantizer->qscheme() == kPerTensorAffine);
  return Scalar(static_cast<PerTensorAffineQuantizer*>(quantizer.get())->scale());
}

Scalar q_zero_point_quant(const Tensor& self) {
  auto quantizer = get_qtensorimpl(self)->quantizer();
  AT_ASSERT(quantizer->qscheme() == kPerTensorAffine);
  return Scalar(static_cast<PerTensorAffineQuantizer*>(quantizer.get())->zero_point());
}

Quantizer* quantizer(const Tensor& self) {
  return get_qtensorimpl(self)->quantizer().get();
}

Tensor int_repr_quant(const Tensor& self) {
  Tensor dst = at::empty(self.sizes(), self.options().dtype(at::kByte));
  uint8_t* self_data = reinterpret_cast<uint8_t *>(self.data<quint8>());
  uint8_t* dst_data = dst.data<uint8_t>();
  memcpy(dst_data, self_data, self.numel());
  return dst;
}

} // namespace native
} // namespace at
