#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/SortingUtils.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/topk_native.h>
#endif

namespace at {
namespace native {

// Currently internal-only.
//
// This implementation assumes the quantizer for the input and the out-
// put are the same.
//
// If we want to support this publicly, we need to add
// a requantization step to the kernel.
std::tuple<Tensor&, Tensor&> quantized_topk_out_cpu(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");
  _allocate_or_resize_output_with_indices(values, indices, self, dim_, k);

  qtopk_stub(kCPU, values, indices, self, k, dim, largest, sorted);

  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> topk_quantized_cpu(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  auto qscheme = self.qscheme();
  TORCH_CHECK(
      qscheme == QScheme::PER_TENSOR_AFFINE ||
          qscheme == QScheme::PER_TENSOR_SYMMETRIC,
      "Top-K is only supported on per-tensor quantization");
  Tensor values = at::_empty_affine_quantized(
    {0},
    self.options(),
    self.q_scale(),
    self.q_zero_point());
  Tensor indices = at::empty({0}, self.options().dtype(kLong));
  return quantized_topk_out_cpu(values, indices, self, k, dim, largest, sorted);
}

DEFINE_DISPATCH(qtopk_stub);

}}  // namespace at::native
