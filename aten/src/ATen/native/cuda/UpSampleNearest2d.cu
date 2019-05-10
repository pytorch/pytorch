#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCUDA.h>

namespace at {
namespace native {

Tensor& upsample_nearest2d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
    return at::legacy::cuda::_thnn_upsample_nearest2d_forward_out(
        output, input, output_size);
}

Tensor upsample_nearest2d_cuda(
    const Tensor& input,
    IntArrayRef output_size) {
    return at::legacy::cuda::_thnn_upsample_nearest2d_forward(
        input, output_size);
}

Tensor& upsample_nearest2d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
    return at::legacy::cuda::_thnn_upsample_nearest2d_backward_out(
        grad_input, grad_output, output_size, input_size);
}

Tensor upsample_nearest2d_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
    return at::legacy::cuda::_thnn_upsample_nearest2d_backward(
        grad_output, output_size, input_size);
}

} // native
} // at
