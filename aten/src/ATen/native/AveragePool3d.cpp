#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ScalarOps.h>
#include <ATen/Parallel.h>
#include <ATen/native/Pool.h>
#include <c10/util/irange.h>
#include <iostream>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/avg_pool3d_backward_native.h>
#include <ATen/ops/avg_pool3d_native.h>
#endif

namespace at::meta {
using namespace ::at::native;

TORCH_META_FUNC(avg_pool3d) (
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  std::optional<int64_t> divisor_override
) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kD : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kD : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
    "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
    "divisor must be not zero");

  /* sizes */
  const int64_t nbatch = input.size(0);
  const int64_t nslices = input.size(-4);
  const int64_t idepth = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t odepth = pooling_output_shape<int64_t>(idepth, kD, padD, dD, 1, ceil_mode);
  const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  pool3d_shape_check(
    input,
    nslices,
    kD, kH, kW,
    dD, dH, dW,
    padD, padH, padW,
    1, 1, 1,
    idepth, iheight, iwidth,
    odepth, oheight, owidth,
    "avg_pool3d()",
    /*check_input_size=*/ true);

  /* resize output */
  if (input.ndimension() == 4) {
    set_output_raw_strided(0, {nslices, odepth, oheight, owidth}, {}, input.options());
  }
  else {
    set_output_raw_strided(0, {nbatch, nslices, odepth, oheight, owidth}, {}, input.options());
  }
}

TORCH_META_FUNC(avg_pool3d_backward) (
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  std::optional<int64_t> divisor_override
) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kD : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kD : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
    "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero");

  const int64_t nslices = input.size(-4);
  const int64_t idepth = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  /* XXX shape check behavior from TH */
  const int64_t odepth_for_shape_check = pooling_output_shape<int64_t>(idepth, kD, padD, dD, 1, ceil_mode);
  const int64_t oheight_for_shape_check = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth_for_shape_check = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  avg_pool3d_backward_shape_check(
    input,
    gradOutput_,
    nslices,
    kD, kH, kW,
    dD, dH, dW,
    padD, padH, padW,
    idepth, iheight, iwidth,
    odepth_for_shape_check, oheight_for_shape_check, owidth_for_shape_check,
    "avg_pool3d_backward()");

  /* resize output */
  set_output_raw_strided(0, input.sizes(), {}, input.options());
}

} // namespace at::meta

namespace at::native {

TORCH_IMPL_FUNC(avg_pool3d_out_cpu)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& output) {
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kD : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kD : safe_downcast<int, int64_t>(kernel_size[2]);

  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[2]);

  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);

  avg_pool3d_kernel(
      kCPU,
      output,
      input,
      kW,
      kH,
      kD,
      dW,
      dH,
      dD,
      padW,
      padH,
      padD,
      count_include_pad,
      divisor_override);
}

TORCH_IMPL_FUNC(avg_pool3d_backward_out_cpu) (
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  std::optional<int64_t> divisor_override,
  const Tensor& gradInput
) {
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kD : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kD : safe_downcast<int, int64_t>(kernel_size[2]);

  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[2]);

  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);

  /* get contiguous gradOutput */
  Tensor gradOutput = gradOutput_.contiguous();

  gradInput.zero_();

  avg_pool3d_backward_kernel(
      kCPU, gradInput, gradOutput,
      kW, kH, kD, dW, dH, dD, padW, padH, padD,
      count_include_pad, divisor_override);
}

DEFINE_DISPATCH(avg_pool3d_kernel);
DEFINE_DISPATCH(avg_pool3d_backward_kernel);

} // namespace at::native
