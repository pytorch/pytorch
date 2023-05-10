#include <ATen/core/Tensor.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor> attn(const Tensor& query, const Tensor& key, const Tensor &value) {
    auto attn = at::matmul(query, key.transpose(-2, -1);
    attn = at::softmax(attn, -1);
    return std::make_tuple(at::matmul(attn, value), attn);
}

} // namespace native
} // namespace at
