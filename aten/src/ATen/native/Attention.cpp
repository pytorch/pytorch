#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <c10/util/Exception.h>

namespace at {
namespace native {

std::tuple <Tensor, Tensor> attn(const Tensor& q, const Tensor& k, const Tensor& v) {
    Tensor k_t = k.t();
    Tensor mm_output = at::native::matmul(q, k_t);
    Tensor tanh_output = at::tanh(mm_output);
    Tensor output = at::native::matmul(tanh_output, v);
    return std::make_tuple(output, tanh_output);
}

}} // namespace at::native
