#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctions.h>

namespace at {
namespace native {

Tensor& upsample_linear1d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners) {
    return at::legacy::th::_thnn_upsample_linear1d_forward_out(
        output, input, output_size, align_corners);
}

Tensor upsample_linear1d_cuda(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners) {
    auto output = at::empty({0}, input.options());
    return at::legacy::th::_thnn_upsample_linear1d_forward(
        output, input, output_size, align_corners);
}

Tensor& upsample_linear1d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners) {
        return at::legacy::th::_thnn_upsample_linear1d_backward_out(
        grad_input, grad_output, output_size, input_size, align_corners);
}

Tensor upsample_linear1d_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners) {
    auto grad_input = at::zeros_like(grad_output);
    return at::legacy::th::_thnn_upsample_linear1d_backward(
        grad_input, grad_output, output_size, input_size, align_corners);
}

namespace {

} // namespace

} // native
} // at