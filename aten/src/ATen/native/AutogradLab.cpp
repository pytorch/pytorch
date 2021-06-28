#include <ATen/ATen.h>
#include <ATen/core/DimVector.h>
#include <c10/util/SmallBuffer.h>

#include <vector>

namespace at {
namespace native {

std::tuple<Tensor, Tensor> attn(const Tensor& q, const Tensor& k, const Tensor& v) {
    Tensor attn = tanh(q.matmul(k.swapaxes(0, 1)));
    Tensor output = attn.matmul(v);
    return std::make_tuple(output, attn);
}

}} // at::native
