#include <ATen/ATen.h>

namespace at {
namespace native {
std::tuple<Tensor, Tensor> attn(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v) {
  auto x = at::mm(q, at::transpose(k, 0, 1));
  auto a = at::tanh(x);
  auto o = at::mm(a, v);
  return std::tuple<Tensor, Tensor>(o, a);
}

Tensor attn_q_backward(
    const Tensor& grad_o,
    const Tensor& grad_a,
    const Tensor& a,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v) {
  Tensor grad_q = at::zeros_like(q);
  if (grad_o.defined()) {
    auto intermediate = at::matmul(grad_o, at::transpose(v, 0, 1));
    grad_q += at::matmul(intermediate * (1 - pow(a, 2)), k);
  }
  if (grad_a.defined()) {
    grad_q += at::matmul(grad_a * (1 - pow(a, 2)), k);
  }
  return grad_q;
}

Tensor attn_k_backward(
    const Tensor& grad_o,
    const Tensor& grad_a,
    const Tensor& a,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v) {
  Tensor grad_k = at::zeros_like(q);
  if (grad_o.defined()) {
    auto intermediate = at::matmul(grad_o, at::transpose(v, 0, 1)) * (1 - pow(a, 2));
    grad_k += at::matmul(at::transpose(intermediate, 0, 1), q);
  }
  if (grad_a.defined()) {
    grad_k += at::matmul(at::transpose(grad_a * (1 - pow(a, 2)), 0, 1), q);
  }
  return grad_k;
  }

Tensor attn_v_backward(const Tensor& grad_o, const Tensor& a, const Tensor& v) {
  Tensor grad_v = at::zeros_like(v);
  if (grad_o.defined()) {
    grad_v += at::matmul(at::transpose(a, 0, 1), grad_o);
  }
  return grad_v;
}
}} // namespace at::native
