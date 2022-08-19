#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/SpmmReduce.h>

namespace at { namespace native {

Tensor spmm_sum_cpu(
    const Tensor& rowptr,
    const Tensor& col,
    const c10::optional<Tensor>& optional_value,
    const Tensor& mat) {
  TORCH_CHECK(rowptr.dim() == 1);
  TORCH_CHECK(col.dim() == 1);
  if (optional_value.has_value()) {
    TORCH_CHECK(optional_value.value().dim() == 1);
    TORCH_CHECK(optional_value.value().size(0) == col.size(0));
  }
  TORCH_CHECK(mat.dim() >= 2);

  Tensor other = mat.contiguous();

  auto sizes = other.sizes().vec();
  sizes[other.dim() - 2] = rowptr.numel() - 1;
  Tensor result = at::empty(sizes, other.options());
  spmm_sum_stub(kCPU, result, rowptr, col, optional_value, other);

  return result;
}

DEFINE_DISPATCH(spmm_sum_stub);

}} // at::native
