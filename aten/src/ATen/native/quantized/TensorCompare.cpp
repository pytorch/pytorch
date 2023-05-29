#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorCompare.h>
#include <c10/util/Exception.h>

namespace at {
namespace native {

Tensor max_quantized_cpu(const Tensor& self) {
  return std::get<0>(self.reshape({-1}).max(/*dim=*/0));
}

Tensor& max_quantized_unary_out(const Tensor& self, Tensor& out) {
  // TODO this implementation is inefficient for now.
  TORCH_CHECK(self.device() == out.device());

  TORCH_CHECK(canCast(
      typeMetaToScalarType(self.dtype()),
      typeMetaToScalarType(out.dtype())));
  Tensor temp = max_quantized_cpu(self);
  at::native::resize_output(out, temp.sizes());
  out.copy_(temp);
  return out;
}

Tensor min_quantized_cpu(const Tensor& self) {
  return std::get<0>(self.reshape({-1}).min(/*dim=*/0));
}

Tensor& min_quantized_unary_out(const Tensor& self, Tensor& out) {
  // TODO this implementation is inefficient for now.
  TORCH_CHECK(self.device() == out.device());

  TORCH_CHECK(canCast(
      typeMetaToScalarType(self.dtype()),
      typeMetaToScalarType(out.dtype())));
  Tensor temp = min_quantized_cpu(self);
  at::native::resize_output(out, temp.sizes());
  out.copy_(temp);
  return out;
}

// TODO: move to TensorMath.cpp

std::tuple<Tensor, Tensor> sort_quantized_cpu_stable(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim,
    bool descending) {
  Tensor sort_int;
  Tensor sort_indicies;
  std::tie(sort_int, sort_indicies) =
      at::sort(self.int_repr(), stable, dim, descending);
  return std::forward_as_tuple(
      at::_make_per_tensor_quantized_tensor(
          sort_int, self.q_scale(), self.q_zero_point()),
      sort_indicies);
}

} // namespace native
} // namespace at
