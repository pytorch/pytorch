#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/xnnpack/Engine.h>
#include <c10/core/GradMode.h>
#include <c10/util/Exception.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/Utils.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_avg_pool1d_native.h>
#include <ATen/ops/adaptive_avg_pool2d.h>
#include <ATen/ops/adaptive_max_pool1d_native.h>
#include <ATen/ops/adaptive_max_pool2d.h>
#include <ATen/ops/avg_pool1d_native.h>
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/max_pool1d_with_indices_native.h>
#include <ATen/ops/max_pool2d_native.h>
#include <ATen/ops/max_pool2d_with_indices.h>
#include <ATen/ops/max_pool3d_native.h>
#include <ATen/ops/max_pool3d_with_indices.h>
#include <ATen/ops/mkldnn_max_pool2d.h>
#include <ATen/ops/mkldnn_max_pool3d.h>
#include <ATen/ops/quantized_max_pool2d.h>
#include <ATen/ops/quantized_max_pool3d.h>
#endif

#include <tuple>

namespace at::native {

static void check1d(
    const char* function_name,
    const char* argument_name,
    IntArrayRef x) {
  TORCH_CHECK(
      x.size() == 1,
      function_name, "() argument '", argument_name,
      "' should contain one int (got ", x.size(), ")");
}

Tensor adaptive_avg_pool1d(const Tensor & self, IntArrayRef output_size) {
  checkDimRange("adaptive_avg_pool1d", TensorArg(self, "self", 1), 2, 4 /* exclusive */);
  check1d("adaptive_avg_pool1d", "output_size", output_size);

  auto output = at::adaptive_avg_pool2d(
      self.unsqueeze(-2),
      {1, output_size[0]});

  return output.squeeze(-2);
}

std::tuple<Tensor,Tensor> adaptive_max_pool1d(const Tensor & self, IntArrayRef output_size) {
  checkDimRange("adaptive_max_pool1d", TensorArg(self, "self", 1), 2, 4 /* exclusive */);
  check1d("adaptive_max_pool1d", "output_size", output_size);

  int ndim = self.ndimension();
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        self.sym_size(i) > 0,
        "adaptive_max_pool1d(): ",
        "Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        self.sym_sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  auto [output, indices] = at::adaptive_max_pool2d(
      self.unsqueeze(-2),
      {1, output_size[0]});

  return std::make_tuple(output.squeeze(-2), indices.squeeze(-2));
}

std::tuple<Tensor, Tensor> max_pool1d_with_indices(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (stride.empty()) {
    stride = kernel_size;
  }
  checkDimRange("max_pool1d", TensorArg(self, "self", 1), 2, 4 /* exclusive */);
  check1d("max_pool1d", "kernel_size", kernel_size);
  check1d("max_pool1d", "stride", stride);
  check1d("max_pool1d", "padding", padding);
  check1d("max_pool1d", "dilation", dilation);

  NoNamesGuard guard;

  auto [output, indices] = at::max_pool2d_with_indices(
      self.unsqueeze(-2),
      {1, kernel_size[0]},
      {1, stride[0]},
      {0, padding[0]},
      {1, dilation[0]},
      ceil_mode);

  output  = output.squeeze(-2);
  indices = indices.squeeze(-2);

  guard.reset();
  namedinference::propagate_names(output, self);
  namedinference::propagate_names(indices, self);

  return std::make_tuple(output, indices);
}

#if AT_MKLDNN_ENABLED()
static bool use_mkldnn_maxpool(const Tensor& input, IntArrayRef dilation, bool is_3d) {
  if (!at::globalContext().userEnabledMkldnn()) {
    return false;
  }
  if (input.is_mkldnn()) {
    return true;
  }
  if (input.sym_numel() <= 1) {
    return false;
  }
  if (input.dim() != (is_3d ? 5 : 4)) {
    return false;
  }
  // Does not support dilation case for now
  // TODO: Add support for dilation
  if (!std::all_of(dilation.cbegin(), dilation.cend(), [](int64_t i) {
        return 1 == i;
      })) {
    return false;
  }
  if (!((GradMode::is_enabled() && input.requires_grad()) ||
        input._fw_grad(/*level */ 0).defined())) {
    bool is_channels_last = is_3d
        ? (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d)
        : (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast);
    if (is_3d) {
      return input.device().is_cpu() && is_channels_last &&
          (((input.scalar_type() == kBFloat16) && mkldnn_bf16_device_check()) ||
           ((input.scalar_type() == kHalf) && mkldnn_fp16_device_check()));
    } else {
      return input.device().is_cpu() &&
          (((input.scalar_type() == kBFloat16) && mkldnn_bf16_device_check()) ||
           ((input.scalar_type() == kHalf) && mkldnn_fp16_device_check()));
    }
  }
  return false;
}
#endif

Tensor avg_pool1d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  if (stride.empty()) {
    stride = kernel_size;
  }
  checkDimRange("avg_pool1d", TensorArg(self, "self", 1), 2, 4 /* exclusive */);
  check1d("avg_pool1d", "kernel_size", kernel_size);
  check1d("avg_pool1d", "stride", stride);
  check1d("avg_pool1d", "padding", padding);

  auto output = at::avg_pool2d(
      self.unsqueeze(-2),
      {1, kernel_size[0]},
      {1, stride[0]},
      {0, padding[0]},
      ceil_mode,
      count_include_pad);

  return output.squeeze(-2);
}

Tensor max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (self.is_quantized()) {
    return at::quantized_max_pool2d(self, kernel_size, stride, padding,
                                    dilation, ceil_mode);
  }
#if defined(C10_MOBILE)
  if(xnnpack::use_max_pool2d(self, kernel_size, padding, stride,
                             dilation, ceil_mode)) {
    return xnnpack::max_pool2d(
        self, kernel_size, padding, stride, dilation, ceil_mode);
  }
#endif

#if AT_MKLDNN_ENABLED()
  // Use mkldnn_max_pool2d to get better performance
  if (use_mkldnn_maxpool(self, dilation, /*is_3d*/ false)) {
    return at::mkldnn_max_pool2d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
#endif

  auto output_and_indices = at::max_pool2d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
  return std::get<0>(output_and_indices);
}

Tensor max_pool3d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (self.is_quantized()) {
    return at::quantized_max_pool3d(self, kernel_size, stride, padding,
                                    dilation, ceil_mode);
  }

#if AT_MKLDNN_ENABLED()
  // Use mkldnn_max_pool3d to get better performance
  if (use_mkldnn_maxpool(self, dilation, /*is_3d*/ true)) {
    return at::mkldnn_max_pool3d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
#endif

  auto output_and_indices = at::max_pool3d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
  return std::get<0>(output_and_indices);
}

} // namespace at::native
