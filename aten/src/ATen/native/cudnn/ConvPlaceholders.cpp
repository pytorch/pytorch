#include <ATen/ATen.h>
#include <ATen/cuda/CUDAConfig.h> // for the definition of AT_CUDNN_ENABLED
#include <ATen/native/ConvUtils.h>

namespace at { namespace native {

// ---------------------------------------------------------------------
//
// Placeholder operators
//
// ---------------------------------------------------------------------

#if !AT_CUDNN_ENABLED()

// See Note [ATen preprocessor philosophy]

at::Tensor cudnn_convolution(
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("cudnn_convolution: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("cudnn_convolution_backward_input: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("cudnn_convolution_backward_weight: ATen not compiled with cuDNN support");
}

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask) {
  AT_ERROR("cudnn_convolution_backward: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_transpose(
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("cudnn_convolution_transpose: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_transpose_backward_input(
    const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("cudnn_convolution_transpose_backward: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_transpose_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("cudnn_convolution_transpose_backward_weight: ATen not compiled with cuDNN support");
}

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_transpose_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask) {
  AT_ERROR("cudnn_convolution_transpose_backward: ATen not compiled with cuDNN support");
}

void raw_cudnn_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("raw_cudnn_convolution_forward_out: ATen not compiled with cuDNN support");
}

void raw_cudnn_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("raw_cudnn_convolution_backward_input_out: ATen not compiled with cuDNN support");
}

void raw_cudnn_convolution_backward_weight_out(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  AT_ERROR("raw_cudnn_convolution_backward_weight_out: ATen not compiled with cuDNN support");
}

Tensor cudnn_convolution_relu(
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_t,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  AT_ERROR("cudnn_convolution_relu: ATen not compiled with cuDNN support");
}

Tensor cudnn_convolution_add_relu(
    const Tensor& input_t,
    const Tensor& weight_t,
    const Tensor& z_t,
    const c10::optional<Scalar>& alpha,
    const c10::optional<Tensor>& bias_t,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  AT_ERROR("cudnn_convolution_add_relu: ATen not compiled with cuDNN support");
}

#endif  // AT_CUDNN_ENABLED

// ---------------------------------------------------------------------
//
// Deprecated operators
//
// ---------------------------------------------------------------------

// TODO (@zasdfgbnm): this is here only for compatibility, remove this in the future
Tensor cudnn_convolution_deprecated(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias /* optional */,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  auto output = at::cudnn_convolution(input, weight, padding, stride, dilation, groups, benchmark, deterministic);
  if (bias.defined()) {
    output = output + reshape_bias(input.dim(), bias);
  }
  return output;
}

// TODO (@zasdfgbnm): this is here only for compatibility, remove this in the future
Tensor cudnn_convolution_deprecated2(
    const Tensor& input_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  return at::cudnn_convolution(input_t, weight_t, padding, stride, dilation, groups, benchmark, deterministic, at::globalContext().allowTF32CuDNN());
}

// TODO (@zasdfgbnm): this is here only for compatibility, remove this in the future
Tensor cudnn_convolution_transpose_deprecated(
    const Tensor& input, const Tensor& weight, const Tensor& bias /* optional */,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  auto output = at::cudnn_convolution_transpose(input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  if (bias.defined()) {
    output = output + reshape_bias(input.dim(), bias);
  }
  return output;
}

// TODO (@zasdfgbnm): this is here only for compatibility, remove this in the future
Tensor cudnn_convolution_transpose_deprecated2(
    const Tensor& input_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
    return at::cudnn_convolution_transpose(input_t, weight_t, padding, output_padding, stride, dilation, groups, benchmark, deterministic, at::globalContext().allowTF32CuDNN());
}

}}
