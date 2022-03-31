#include <ATen/ATen.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor> attn(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v) {

  TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "All of q,k,v need to be 2D tensor");
  TORCH_CHECK(q.size(1) == k.size(1), "q and k must have must same size for 2nd dim");
  TORCH_CHECK(k.size(0) == v.size(0), "k and v must have must same size for 1st dim");

  Tensor x = at::matmul(q, k.t());
  Tensor a = at::tanh(x);
  Tensor o = at::matmul(a, v);

  return std::make_tuple(o, a);
}

std::tuple<Tensor, Tensor, Tensor> attn_backward(
    const Tensor& do_ext,
    const Tensor& da_ext,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& a) {

  Tensor dq, dk, dv;

  bool needs_dq = q.requires_grad();
  bool needs_dk = k.requires_grad();
  bool needs_dv = v.requires_grad();

  // TORCH_CHECK(needs_dq || needs_dk || needs_dv, "At least one of the qkv inputs needs gradient, "
  //                                               "otherwise attn_backward should not be invoked");

  Tensor da_partial;
  if (do_ext.defined()) {
    if (needs_dv) {
      dv = at::matmul(a.t(), do_ext);
    }

    if (needs_dq || needs_dk) {
      da_partial = at::matmul(do_ext, v.t());
    }
  }

  if (needs_dq || needs_dk) {
    Tensor da;
    if (da_ext.defined() && da_partial.defined()) {
      da = da_ext + da_partial;
    } else if (da_ext.defined()) {
      da = da_ext;
    } else if (da_partial.defined()) {
      da = da_partial;
    } else {
      TORCH_CHECK(false, "To compute dq or dk, da should be back-probagated from at least one branch");
    }

    // forward: a = tanh(x)
    // backward: dx = da * (1 - a * a)
    Tensor dx = da * (1 - a * a);

    // forward: x = mm(q, k')
    // backward: dq = mm(dx, k); dk = mm(dx', q)
    if (needs_dq) {
      dq = at::matmul(dx, k);
    }
    if (needs_dk) {
      dk = at::matmul(dx.t(), q);
    }
  }

  return std::make_tuple(dq, dk, dv);
}

} // namespace native
} // namespace at
