#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor adaptive_avg_pool2d(
    const at::Tensor& self_arg,
    const IntArrayRef output_size) {
  TORCH_INTERNAL_ASSERT(
      self_arg.dim() == 4,
      "vulkan_adaptive_avg_pool2d expects 4-dimensional input!");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
    context,
    {
      self.size(Layout::Activation4D::batch),
      self.size(Layout::Activation4D::channels),
      output_size[Layout::Activation4D::batch],
      output_size[Layout::Activation4D::channels],
    },
    self.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_self.has_image()) {
      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          },
          VK_KERNEL(adaptive_avg_pool2d),
          v_output.extents(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(command_buffer, vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_self.image(command_buffer));
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

  return convert(v_output);
}

Tensor avg_pool2d(
    const Tensor& self_arg,
    const IntArrayRef kernel_arg,
    IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const bool ceil_mode,
    const bool /* count_include_pad */,
    const c10::optional<int64_t> /* divisor_override */) {
  if (stride_arg.empty()) {
    stride_arg = kernel_arg;
  }

  TORCH_CHECK(!kernel_arg.empty(), "Kernel size cannot be empty!");
  TORCH_CHECK(!stride_arg.empty(), "Stride cannot be empty!");
  TORCH_CHECK(!padding_arg.empty(), "Padding cannot be empty!");

  static const auto normalize = [](const IntArrayRef parameter) {
    return std::array<int64_t, 2>{
      parameter[0],
      (2 == parameter.size()) ? parameter[1] : parameter[0],
    };
  };

  const auto input_size = self_arg.sizes();
  const auto kernel = normalize(kernel_arg);
  const auto stride = normalize(stride_arg);
  const auto padding = normalize(padding_arg);
  const auto dilation = std::array<int64_t, 2>{1, 1};

  const int64_t output_height = pooling_output_shape(
      input_size[Layout::Activation4D::height],
      kernel[Layout::Parameter::height],
      padding[Layout::Parameter::height],
      stride[Layout::Parameter::height],
      dilation[Layout::Parameter::height],
      ceil_mode);

  const int64_t output_width = pooling_output_shape(
      input_size[Layout::Activation4D::width],
      kernel[Layout::Parameter::width],
      padding[Layout::Parameter::width],
      stride[Layout::Parameter::width],
      dilation[Layout::Parameter::width],
      ceil_mode);

  pool2d_shape_check(
      self_arg,
      kernel[Layout::Parameter::height],
      kernel[Layout::Parameter::width],
      stride[Layout::Parameter::height],
      stride[Layout::Parameter::width],
      padding[Layout::Parameter::height],
      padding[Layout::Parameter::width],
      dilation[Layout::Parameter::height],
      dilation[Layout::Parameter::width],
      input_size[Layout::Activation4D::channels],
      input_size[Layout::Activation4D::height],
      input_size[Layout::Activation4D::width],
      output_height,
      output_width);

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
    context,
    {
      input_size[Layout::Activation4D::batch],
      input_size[Layout::Activation4D::channels],
      output_height,
      output_width,
    },
    self.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    using namespace api::utils;

    if (v_self.has_image()) {
      const struct {
        int32_t kernel_width, kernel_height;
        int32_t stride_x, stride_y;
        int32_t padding_x, padding_y;
      } block {
        safe_downcast<int32_t>(kernel[Layout::Parameter::width]),
        safe_downcast<int32_t>(kernel[Layout::Parameter::height]),
        safe_downcast<int32_t>(stride[Layout::Parameter::width]),
        safe_downcast<int32_t>(stride[Layout::Parameter::height]),
        safe_downcast<int32_t>(padding[Layout::Parameter::width]),
        safe_downcast<int32_t>(padding[Layout::Parameter::height]),
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(avg_pool2d),
          v_output.extents(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(command_buffer, vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_self.image(command_buffer),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("_adaptive_avg_pool2d", TORCH_FN(adaptive_avg_pool2d));
  m.impl("avg_pool2d", TORCH_FN(avg_pool2d));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
