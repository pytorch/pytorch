#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NamedTensorUtils.h>

namespace at {
namespace native {

Tensor &quantized_mean_out_cpu(Tensor &result, const Tensor &self, IntArrayRef dim,
                 bool keepdim, c10::optional<ScalarType> opt_dtype) {
  auto self_dequantized = self.dequantize();
  auto result_dequantized = at::native::mean_cpu_gpu(self_dequantized, dim, keepdim, opt_dtype);
  result = at::quantize_per_tensor(result_dequantized, self.q_scale(), self.q_zero_point(), opt_dtype.value_or(self.scalar_type()));
  return result;
}

Tensor quantized_mean_cpu(const Tensor &self, optional<ScalarType> dtype) {
  Tensor result;
  quantized_mean_out_cpu(result, self, IntArrayRef{}, false, dtype);
  return result;
}

Tensor quantized_mean_cpu(const Tensor& self, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype) {
  Tensor result;
  quantized_mean_out_cpu(result, self, dim, keepdim, dtype);
  return result;
}

#ifdef BUILD_NAMEDTENSOR
Tensor quantized_mean_cpu(const Tensor& self, DimnameList dim, bool keepdim, optional<ScalarType> dtype) {
  return quantized_mean_cpu(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& quantized_mean_out_cpu(Tensor& result, const Tensor& self, DimnameList dim,
                 bool keepdim, c10::optional<ScalarType> opt_dtype) {
  return quantized_mean_out_cpu(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}
#endif


}}  // namespace at::native
