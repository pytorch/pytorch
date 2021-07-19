#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>


namespace at {

namespace meta{
using namespace native;

TORCH_META_FUNC(avg_pool2d) (
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override
) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  kH_ = safe_downcast<int, int64_t>(kernel_size[0]);
  kW_ = kernel_size.size() == 1 ? kH_ : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  dH_ = stride.empty() ? kH_ : safe_downcast<int, int64_t>(stride[0]);
  dW_ = stride.empty() ? kW_ :
                 stride.size() == 1 ? dH_ : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  padH_ = safe_downcast<int, int64_t>(padding[0]);
  padW_ = padding.size() == 1 ? padH_ : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
    "divisor must be not zero");

  /* sizes */
  nbatch_ = input.ndimension() == 4 ? input.size(-4) : 1;
  nInputPlane_ = input.size(-3);
  inputHeight_ = input.size(-2);
  inputWidth_ = input.size(-1);

  outputHeight_ = pooling_output_shape<int64_t>(
      inputHeight_, kH_, padH_, dH_, 1, ceil_mode);
  outputWidth_ =
      pooling_output_shape<int64_t>(inputWidth_, kW_, padW_, dW_, 1, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  pool2d_shape_check(
      input,
      kH_,
      kW_,
      dH_,
      dW_,
      padH_,
      padW_,
      1,
      1,
      nInputPlane_,
      inputHeight_,
      inputWidth_,
      outputHeight_,
      outputWidth_,
      memory_format);

  /* resize output */
  if (input.ndimension() == 3) {
    set_output(0, {nInputPlane_, outputHeight_, outputWidth_}, input.options());
  }
  else {
    set_output(
        0,
        {nbatch_, nInputPlane_, outputHeight_, outputWidth_},
        input.options().memory_format(memory_format));
  }
}

TORCH_META_FUNC(avg_pool2d_backward) (
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override
) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3); // number of channels (or colors)
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  avg_pool2d_backward_shape_check(
    input,
    gradOutput_,
    nbatch,
    kH, kW, dH, dW, padH, padW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth,
    memory_format);

  /* resize output */
  set_output(0, input.sizes(), input.options().memory_format(memory_format));
}

} // namespace meta

namespace native {

TORCH_IMPL_FUNC(avg_pool2d_out_cpu) (
  const Tensor &input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override,
  const Tensor &output
) {
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  avg_pool2d_kernel(
      kCPU, output, input,
      kW, kH, dW, dH, padW, padH,
      count_include_pad, divisor_override);
}

TORCH_IMPL_FUNC(avg_pool2d_backward_out_cpu) (
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  c10::optional<int64_t> divisor_override,
  const Tensor& gradInput
) {
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero");

  TORCH_CHECK(input.dtype() == gradOutput.dtype(),
    "expected dtype ", input.dtype(), " for `gradOutput` but got dtype ", gradOutput.dtype());

  /* zero the gradient */
  gradInput.zero_();

  avg_pool2d_backward_kernel(
      kCPU, gradInput, gradOutput,
      kW, kH, dW, dH, padW, padH,
      count_include_pad, divisor_override);
}

DEFINE_DISPATCH(avg_pool2d_kernel);
DEFINE_DISPATCH(avg_pool2d_backward_kernel);

} // at::native
} // at
