#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/quantized/QTensorImpl.h>


namespace at {
namespace native {

Tensor quantize_linear_cpu(const Tensor& self, double scale, int64_t zero_point, ScalarType dtype) {
  auto quantizer = make_per_tensor_affine_quantizer(scale, zero_point, dtype);
  return quantizer->quantize(self);
}

Tensor dequantize_quant(const Tensor& self) {
  return get_qtensorimpl(self)->quantizer()->dequantize(self);
}

Tensor dequantize_linear_cpu(const Tensor& self, double scale, int64_t zero_point, ScalarType dtype) {
  AT_CHECK(isQIntType(toQIntType(self.scalar_type())),
           "Scalar type for quantized Tensor must have same underlying type as input.");
  AT_CHECK(dtype == ScalarType::Float, "ScalarType for target Tensor must be float.");
  Tensor f = at::empty(self.sizes(), self.options().dtype(dtype));
  AT_DISPATCH_QINT_TYPES(
      toQIntType(self.scalar_type()), "dequantize_linear_cpu", [&]() {
        underlying_t* qdata = self.data<underlying_t>();
        auto* fdata = f.data<float>();
        for (int i = 0; i < self.numel(); ++i) {
          fdata[i] = (static_cast<float>(qdata[i]) - zero_point) * scale;
        }});
  return f;
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
  Tensor dst;
  AT_DISPATCH_QINT_TYPES(
      self.scalar_type(), "int_repr", [&]() {
        dst = at::empty(self.sizes(), self.options().dtype(UNDERLYING_TYPE));
        underlying_t* self_data = reinterpret_cast<underlying_t *>(self.data<scalar_t>());
        underlying_t* dst_data = dst.data<underlying_t>();
        if (self.numel() > 0) {
          memcpy(dst_data, self_data, self.nbytes());
        }});
  return dst;
}

Tensor per_tensor_affine_qtensor_cpu(const Tensor& self, double scale, int64_t zero_point) {
  Tensor dst = at::_empty_affine_quantized(self.sizes(), self.options().dtype(toQIntType(self.scalar_type())), scale, zero_point);
  AT_DISPATCH_QINT_TYPES(dst.scalar_type(), "per_tensor_affine_qtensor", [&]() {
    underlying_t* self_data = self.data<underlying_t>();
    underlying_t* dst_data = reinterpret_cast<underlying_t *>(dst.data<scalar_t>());
    if (self.numel() > 0) {
      memcpy(dst_data, self_data, self.numel());
    }
  });
  return dst;
}

} // namespace native
} // namespace at
