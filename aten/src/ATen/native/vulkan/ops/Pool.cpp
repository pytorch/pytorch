#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor adaptive_avg_pool2d(
    const at::Tensor& self_arg,
    const IntArrayRef output_size) {
  TORCH_CHECK(
      self_arg.dim() == 4,
      "Vulkan adaptive_avg_pool2d expects 4-dimensional input!");

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
    v_self.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    if C10_LIKELY(v_self.has_image()) {
      const uvec3 v_output_size = v_output.extents();
      const uvec3 v_self_size = v_self.extents();

      const vec2 stride {
        static_cast<float>(v_self_size.data[0u]) / v_output_size.data[0u],
        static_cast<float>(v_self_size.data[1u]) / v_output_size.data[1u],
      };

      const struct Block final {
        uvec3 extents;
        uint32_t _;
        vec2 kernel;
        vec2 stride;
      } block {
        v_output.extents(),
        0u,
        {
          v_self_size.data[0u] - (v_output_size.data[0u] - 1u) * stride.data[0u],
          v_self_size.data[1u] - (v_output_size.data[1u] - 1u) * stride.data[1u],
        },
        stride,
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(adaptive_avg_pool2d),
          v_output.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_self.image(
              command_buffer,
              vTensor::Stage::Compute),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

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
      output_width,
      self_arg.suggest_memory_format());

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
    v_self.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    if C10_LIKELY(v_self.has_image()) {
      const struct Block final {
        uvec3 extents;
        int32_t range;
        ivec4 kernel;
        ivec2 stride;
        ivec2 padding;
      } block {
        v_output.extents(),
        safe_downcast<int32_t>(
            kernel[Layout::Parameter::width] *
            kernel[Layout::Parameter::height]),
        {
          safe_downcast<int32_t>(kernel[Layout::Parameter::width]),
          safe_downcast<int32_t>(kernel[Layout::Parameter::height]),
          safe_downcast<int32_t>(self.size(Layout::Activation4D::width)),
          safe_downcast<int32_t>(self.size(Layout::Activation4D::height)),
        },
        {
          safe_downcast<int32_t>(stride[Layout::Parameter::width]),
          safe_downcast<int32_t>(stride[Layout::Parameter::height]),
        },
        {
          safe_downcast<int32_t>(padding[Layout::Parameter::width]),
          safe_downcast<int32_t>(padding[Layout::Parameter::height]),
        },
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
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_self.image(
              command_buffer,
              vTensor::Stage::Compute),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

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
