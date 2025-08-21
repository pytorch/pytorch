#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/xnnpack/Engine.h>
#include <c10/util/Exception.h>

#include <ATen/ops/ones_like_native.h>

namespace at::native {

std::tuple<Tensor, Tensor, Tensor>
lab_attn_bwd(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& a, const Tensor& grad_o, const Tensor&) {
  auto grad_v = a.t().matmul(grad_o);
  auto grad_a = grad_o.matmul(V.t());

  auto one = at::native::ones_like(a);
  auto grad_x = grad_a.mul(one.sub(a.mul(a)));

  auto grad_q = grad_x.matmul(K);
  auto grad_k = ((Q.t()).matmul(grad_x)).t();

  return std::make_tuple(grad_q, grad_k, grad_v);
}

std::tuple<Tensor,Tensor> lab_attn(const Tensor& Q, const Tensor& K, const Tensor& V) {
  auto x = Q.matmul(K.t());
  auto a = x.tanh();
  auto o = a.matmul(V);

  return std::make_tuple(o, a);
}

};  // at::native namespace
