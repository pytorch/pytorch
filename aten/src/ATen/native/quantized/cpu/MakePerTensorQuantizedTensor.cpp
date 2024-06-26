#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_make_per_tensor_quantized_tensor_native.h>
#endif

namespace at {
namespace native {

Tensor make_per_tensor_quantized_tensor_cpu(
    const Tensor& self,
    double scale,
    int64_t zero_point) {
  Tensor dst = at::_empty_affine_quantized(
      self.sizes(),
      self.options().dtype(toQIntType(self.scalar_type())),
      scale,
      zero_point,
      self.suggest_memory_format());
  Tensor self_contig = self.contiguous(self.suggest_memory_format());
  AT_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "make_per_tensor_quantized_tensor", [&]() {
        underlying_t* self_data = self_contig.data_ptr<underlying_t>();
        underlying_t* dst_data =
            reinterpret_cast<underlying_t*>(dst.data_ptr<scalar_t>());
        if (self.numel() > 0) {
          memcpy(dst_data, self_data, self.nbytes());
        }
      });
  return dst;
}

} // namespace native
} // namespace at
