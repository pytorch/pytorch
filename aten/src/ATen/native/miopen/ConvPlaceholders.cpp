#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_ROCM_ENABLED
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/miopen_convolution_add_relu_native.h>
#include <ATen/ops/miopen_convolution_native.h>
#include <ATen/ops/miopen_convolution_relu_native.h>
#include <ATen/ops/miopen_convolution_transpose_native.h>
#include <ATen/ops/miopen_depthwise_convolution_native.h>
#endif

namespace at::native {

// ---------------------------------------------------------------------
//
// Placeholder operators
//
// ---------------------------------------------------------------------

#if !AT_ROCM_ENABLED()

// See Note [ATen preprocessor philosophy]

at::Tensor miopen_convolution(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  TORCH_CHECK(
      false,
      "miopen_convolution: ATen not compiled with MIOpen support");
}

at::Tensor& miopen_convolution_out(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    Tensor& output_t) {
  TORCH_CHECK(
      false,
      "miopen_convolution_out: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_backward_input(
    IntArrayRef input_size,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  TORCH_CHECK(
      false,
      "miopen_convolution_backward_input: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_backward_weight(
    IntArrayRef weight_size,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  TORCH_CHECK(
      false,
      "miopen_convolution_backward_weight: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_backward_bias(
    const at::Tensor& grad_output) {
  TORCH_CHECK(
      false,
      "miopen_convolution_backward_bias: ATen not compiled with MIOpen support");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> miopen_convolution_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool,3> output_mask) {
  TORCH_CHECK(
      false,
      "miopen_convolution_backward: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_transpose(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  TORCH_CHECK(
      false,
      "miopen_convolution_transpose: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_transpose_backward_input(
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  TORCH_CHECK(
      false,
      "miopen_convolution_transpose_backward: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_transpose_backward_weight(
    IntArrayRef weight_size,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  TORCH_CHECK(
      false,
      "miopen_convolution_transpose_backward_weight: ATen not compiled with MIOpen support");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> miopen_convolution_transpose_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool,3> output_mask) {
  TORCH_CHECK(
      false,
      "miopen_convolution_transpose_backward: ATen not compiled with MIOpen support");
}

at::Tensor miopen_depthwise_convolution(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  TORCH_CHECK(
      false,
      "miopen_depthwise_convolution: ATen not compiled with MIOpen support");
}

at::Tensor miopen_depthwise_convolution_backward_input(
    IntArrayRef input_size,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  TORCH_CHECK(
      false,
      "miopen_depthwise_convolution_backward_input: ATen not compiled with MIOpen support");
}

at::Tensor miopen_depthwise_convolution_backward_weight(
    IntArrayRef weight_size,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  TORCH_CHECK(
      false,
      "miopen_depthwise_convolution_backward_weight: ATen not compiled with MIOpen support");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> miopen_depthwise_convolution_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool,3> output_mask) {
  TORCH_CHECK(
      false,
      "miopen_depthwise_convolution_backward: ATen not compiled with MIOpen support");
}


at::Tensor miopen_convolution_add_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& z,
    const std::optional<Scalar>& alpha,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(
      false,
      "miopen_convolution_add_relu: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(
      false,
      "miopen_convolution_relu: ATen not compiled with MIOpen support");
}

#endif // AT_ROCM_ENABLED()

} // namespace at::native
