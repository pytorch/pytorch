#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/Pool.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/avg_pool2d_backward_native.h>
#include <ATen/ops/avg_pool2d_native.h>
#endif

namespace at::meta {
using namespace ::at::native;

TORCH_PRECOMPUTE_META_FUNC(avg_pool2d)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  const int64_t kH = kernel_size[0];
  const int64_t kW = kernel_size.size() == 1 ? kH : kernel_size[1];

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int64_t dH = stride.empty() ? kH : stride[0];
  const int64_t dW = stride.empty() ? kW : stride.size() == 1 ? dH : stride[1];

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int64_t padH = padding[0];
  const int64_t padW = padding.size() == 1 ? padH : padding[1];

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
    "divisor must be not zero");

  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  pool2d_shape_check(
      input,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      1,
      1,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  /* resize output */
  if (input.ndimension() == 3) {
    set_output_raw_strided(
        0,
        {nInputPlane,
         outputHeight,
         outputWidth},
        {},
        input.options());
  }
  else {
    set_output_raw_strided(
        0,
        {nbatch,
         nInputPlane,
         outputHeight,
         outputWidth},
        {},
        input.options().memory_format(memory_format));
  }

  return TORCH_PRECOMPUTE_STRUCT(avg_pool2d)().set_kH(kH).set_kW(kW).set_dH(dH).set_dW(dW).set_padH(padH).set_padW(padW);
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
  set_output_raw_strided(0, input.sizes(), {}, input.options().memory_format(memory_format));
}

} // namespace at::meta

namespace at::native {

TORCH_IMPL_FUNC(avg_pool2d_out_cpu)
(const Tensor& input,
 int64_t kH,
 int64_t kW,
 int64_t dH,
 int64_t dW,
 int64_t padH,
 int64_t padW,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override,
 const Tensor& output) {
  avg_pool2d_kernel(
      kCPU,
      output,
      input,
      kW,
      kH,
      dW,
      dH,
      padW,
      padH,
      count_include_pad,
      divisor_override);
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

} // namespace at::native
