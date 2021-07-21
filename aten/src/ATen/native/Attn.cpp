#include <tuple>

#include <ATen/Tensor.h>

namespace at { namespace native {

namespace {

auto is_matrix(Tensor const& t) -> bool {
  return t.dim() == 2;
}

}  // namespace

auto attn(Tensor const& q, Tensor const& k, Tensor const& v) -> std::tuple<Tensor, Tensor> {
  TORCH_CHECK(is_matrix(q), "q is not a matrix: q.shape = ", q.sizes());
  TORCH_CHECK(is_matrix(k), "k is not a matrix: k.shape = ", k.sizes());
  TORCH_CHECK(is_matrix(v), "v is not a matrix: v.shape = ", v.sizes());
  TORCH_CHECK(q.size(0) == k.size(0) && k.size(0) == v.size(0),
              "q, k, and v must all have the same number of rows, but"
              "\n\tq.shape[0]: ", q.size(0),
              "\n\tk.shape[0]: ", k.size(0),
              "\n\tv.shape[0]: ", v.size(0));
  TORCH_CHECK(q.size(1) == k.size(1), "q and k must have the same number of columns, but"
              "\n\tq.shape[1]: ", q.size(1),
              "\n\tk.shape[1]: ", k.size(1));
  Tensor attn = q.matmul(k.t()).tanh();
  Tensor output = attn.matmul(v);
  return std::make_tuple(output, attn);
}

}  // namespace native
}  // namespace at
