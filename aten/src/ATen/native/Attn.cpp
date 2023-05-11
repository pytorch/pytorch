#include <ATen/core/Tensor.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor> attn(const Tensor& query, const Tensor& key, const Tensor &value) {
    TORCH_CHECK(query.dim() == 2, "expected 2D `query`, got ", query.dim(), "-D tensor")
    TORCH_CHECK(key.dim() == 2, "expected 2D `key`, got ", key.dim(), "-D tensor")
    TORCH_CHECK(value.dim() == 2, "expected 2D `value`, got ", value.dim(), "-D tensor")

    TORCH_CHECK(query.sizes()[1] == key.sizes()[1], "expected size of key to be ", query.sizes()[1], " in last dimension but got: ", key.sizes()[1])
    auto attn = at::native::matmul(query, key.transpose(-2, -1));
    attn = at::softmax(attn, -1);
    return std::make_tuple(at::matmul(attn, value), attn);
}

} // namespace native
} // namespace at
