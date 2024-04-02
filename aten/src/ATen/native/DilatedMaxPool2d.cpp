#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/Pool.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/max_pool2d_with_indices_backward_native.h>
#include <ATen/ops/max_pool2d_with_indices_native.h>
#endif

namespace at {
namespace meta {
using namespace native;
TORCH_META_FUNC(max_pool2d_with_indices)
(const Tensor& input,
IntArrayRef kernel_size,
IntArrayRef stride,
IntArrayRef padding,
IntArrayRef dilation,
bool ceil_mode) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(false, "Unsupport memory format. Supports only ChannelsLast, Contiguous");
  }

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

  pool2d_shape_check(
    input,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth, memory_format);

  /* resize output and indices */
  DimnameList maybe_names = input.has_names() ? input.names() : DimnameList{};
  if (input.ndimension() == 3) {
    set_output_raw_strided(0, {nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format), maybe_names);
    /* indices will contain the locations for each output point */
    set_output_raw_strided(1, {nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format).dtype(kLong), maybe_names);
  } else {
    set_output_raw_strided(0, {nbatch, nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format), maybe_names);
    /* indices will contain the locations for each output point */
    set_output_raw_strided(1, {nbatch, nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format).dtype(kLong), maybe_names);
  }
}

TORCH_META_FUNC(max_pool2d_with_indices_backward)
(const Tensor& gradOutput,
const Tensor& input,
IntArrayRef kernel_size,
IntArrayRef stride,
IntArrayRef padding,
IntArrayRef dilation,
bool ceil_mode,
const Tensor& indices) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(input.dtype() == gradOutput.dtype(),
    "expected dtype ", input.dtype(), " for `gradOutput` but got dtype ", gradOutput.dtype());

  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(input.ndimension() == 4,
      "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(false, "Unsupport memory format. Supports only ChannelsLast, Contiguous");
  }

  /* sizes */
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  /* XXX preserve the existing shape check behavior */
  const int64_t outputHeight_for_shape_check = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth_for_shape_check = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

  max_pool2d_backward_shape_check(
    input,
    gradOutput,
    indices,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight_for_shape_check, outputWidth_for_shape_check,
    memory_format);

  set_output_raw_strided(0, input.sizes(), {}, input.options().memory_format(memory_format),
             input.has_names() ? input.names() : DimnameList{});
}
} // namespace meta

namespace native {

TORCH_IMPL_FUNC(max_pool2d_with_indices_out_cpu)
(const Tensor& input,
IntArrayRef kernel_size,
IntArrayRef stride,
IntArrayRef padding,
IntArrayRef dilation,
bool ceil_mode,
const Tensor& output,
const Tensor& indices) {
  NoNamesGuard guard;

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  max_pool2d_kernel(
      kCPU, output, indices, input,
      kW, kH,
      dW, dH,
      padW, padH,
      dilationW, dilationH);
}

TORCH_IMPL_FUNC(max_pool2d_with_indices_backward_out_cpu)
(const Tensor& gradOutput,
const Tensor& input,
IntArrayRef kernel_size,
IntArrayRef stride,
IntArrayRef padding,
IntArrayRef dilation,
bool ceil_mode,
const Tensor& indices,
const Tensor& gradInput) {
  NoNamesGuard guard;

  gradInput.zero_();
  max_pool2d_backward_kernel(
      kCPU, const_cast<Tensor&>(gradInput),
      gradOutput, indices);
}

DEFINE_DISPATCH(max_pool2d_kernel);
DEFINE_DISPATCH(max_pool2d_backward_kernel);

} // at::native
} // at
