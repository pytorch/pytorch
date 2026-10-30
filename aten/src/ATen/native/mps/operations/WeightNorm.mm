#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_weight_norm_differentiable_backward.h>
#include <ATen/ops/_weight_norm_interface_backward_native.h>
#include <ATen/ops/_weight_norm_interface_native.h>
#include <ATen/ops/norm_except_dim.h>
#endif

namespace at::native {

std::tuple<Tensor, Tensor> weight_norm_mps(const Tensor& v, const Tensor& g, int64_t dim) {
  TORCH_CHECK(dim == 0 || dim == v.dim() - 1, "fused kernels can only be applied for first or last dim")
  auto norms = at::norm_except_dim(v, 2, dim);
  return {v.mul(g.div(norms)), norms};
}

std::tuple<Tensor, Tensor> weight_norm_backward_mps(const Tensor& grad_w,
                                                    const Tensor& saved_v,
                                                    const Tensor& saved_g,
                                                    const Tensor& saved_norms,
                                                    int64_t dim) {
  return at::_weight_norm_differentiable_backward(grad_w.contiguous(), saved_v, saved_g, saved_norms, dim);
}

} // namespace at::native
