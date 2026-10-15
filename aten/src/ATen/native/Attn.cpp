// Autograd onboarding lab: attn(q, k, v) = (tanh(q @ k^T) @ v, tanh(q @ k^T)).
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/attn_native.h>
#endif

namespace at::native {

std::tuple<Tensor, Tensor> attn(const Tensor& q, const Tensor& k, const Tensor& v) {
  TORCH_CHECK(
      q.dim() == 2 && k.dim() == 2 && v.dim() == 2,
      "attn: expected q, k and v to be 2D, but got ",
      q.dim(), "D, ", k.dim(), "D and ", v.dim(), "D tensors");
  TORCH_SYM_CHECK(
      q.sym_size(0).sym_eq(k.sym_size(0)) & q.sym_size(0).sym_eq(v.sym_size(0)),
      "attn: expected q, k and v to have the same size in dimension 0, but got ",
      q.sym_size(0), ", ", k.sym_size(0), " and ", v.sym_size(0));
  TORCH_SYM_CHECK(
      q.sym_size(1).sym_eq(k.sym_size(1)),
      "attn: expected q and k to have the same size in dimension 1, but got ",
      q.sym_size(1), " and ", k.sym_size(1));

  auto a = q.matmul(k.transpose(0, 1)).tanh();
  auto o = a.matmul(v);
  return std::make_tuple(std::move(o), std::move(a));
}

} // namespace at::native
