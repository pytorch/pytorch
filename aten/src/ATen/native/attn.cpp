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

} // namespace native
} // namespace at
