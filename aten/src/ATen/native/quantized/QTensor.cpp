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

Tensor quantize_linear_per_channel_cpu(const Tensor& self, const Tensor& scales, const Tensor& zero_points, const Tensor& axis, ScalarType dtype) {
  TORCH_CHECK(scales.dim() == 1, "scale tensor must have dimension 1");
  TORCH_CHECK(zero_points.dim() == 1, "zero_points tensor must have dimension 1");
  TORCH_CHECK(scales.numel() == zero_points.numel(), "number of elements in scales and zero_points must match");
  TORCH_CHECK(axis.dim() == 1, "axis tensor must have dimension 1");
  std::vector<float> scale_vals;
  std::vector<int32_t> zero_point_vals;
  std::vector<int64_t> axis_vals;
  for (auto i = 0; i < scales.numel(); ++i) {
    scale_vals.push_back(scales.data<float>()[i]);
    zero_point_vals.push_back(zero_points.data<int32_t>()[i]);
  }
  for (auto i = 0; i < axis.numel(); ++i) {
    axis_vals.push_back(axis.data<int64_t>()[i]);
  }
  auto quantizer = make_per_channel_affine_quantizer(scale_vals, zero_point_vals, axis_vals, dtype);
  return quantizer->quantize(self);
}

Tensor dequantize_quant(const Tensor& self) {
  return get_qtensorimpl(self)->quantizer()->dequantize(self);
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
  if (self.numel() > 0) {
    memcpy(dst_data, self_data, self.numel());
  }
  return dst;
}

} // namespace native
} // namespace at
