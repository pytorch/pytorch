#include <ATen/ATen.h>
#include <torch/library.h>

namespace at {
namespace native {

Tensor rotate_half(const Tensor& x) {
    auto half_dim = x.size(-1) / 2;
    auto x1 = x.narrow(-1, 0, half_dim);
    auto x2 = x.narrow(-1, half_dim, half_dim);
    return at::cat({-x2, x1}, -1);
}


std::tuple<Tensor, Tensor> rotary_embedding(const Tensor& q, const Tensor& k, const Tensor& cos, const Tensor& sin, const Tensor& position_ids, int64_t unsqueeze_dim) {
    TORCH_CHECK(q.dim() == 4, "rotary_embedding: Expected 4-D argument q but got ", q.dim(), "-D");
    TORCH_CHECK(k.dim() == 4, "rotary_embedding: Expected 4-D argument k but got ", k.dim(), "-D");
    TORCH_CHECK(cos.dim() == 2, "rotary_embedding: Expected 2-D argument cos but got ", cos.dim(), "-D");
    TORCH_CHECK(sin.dim() == 2, "rotary_embedding: Expected 2-D argument sin but got ", sin.dim(), "-D");
    TORCH_CHECK(position_ids.dim() == 1, "rotary_embedding: Expected 1-D argument position_ids but got ", position_ids.dim(), "-D");
    
    auto cos_adjusted = cos.index_select(0, position_ids).unsqueeze(unsqueeze_dim);
    auto sin_adjusted = sin.index_select(0, position_ids).unsqueeze(unsqueeze_dim);
    
    cos_adjusted = cos_adjusted.unsqueeze(0);
    sin_adjusted = sin_adjusted.unsqueeze(0);

    auto q_embed = (q * cos_adjusted) + (rotate_half(q) * sin_adjusted);
    auto k_embed = (k * cos_adjusted) + (rotate_half(k) * sin_adjusted);

    return std::make_tuple(q_embed, k_embed);
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("rotary_embedding", TORCH_FN(rotary_embedding));
}

} // namespace native
} // namespace at
