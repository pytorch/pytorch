#include <ATen/core/Tensor.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> slow_conv3d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    std::array<bool, 3> output_mask);

}} // namespace at::native
