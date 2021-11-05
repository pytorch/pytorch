#include <ATen/ATen.h>
#include <tuple>


namespace at{
namespace native{

std::tuple<Tensor, Tensor> onboarding_attn(const Tensor& q, const Tensor& k, const Tensor& v){
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "Expected all inputs to be 2D");
    TORCH_CHECK(q.size(0) == k.size(0) && k.size(0) == v.size(0), "Expected 0th dimension of all tensors to be the same");
    TORCH_CHECK(q.size(1) == k.size(1), "Expected 1st dimension of first and second input to be the same");
    TORCH_CHECK(!q.is_complex());
    auto x = at::matmul(q, at::transpose(k, 0, 1));
    auto a = at::tanh(x);
    auto o = at::matmul(a, v);
    return std::make_tuple(o, a);
}

}
}
