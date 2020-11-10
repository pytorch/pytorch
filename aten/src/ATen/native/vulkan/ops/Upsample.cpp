#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/UpSample.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor upsample_nearest2d(
    const Tensor& input_arg,
    const IntArrayRef output_sizes,
    const c10::optional<double> scales_h,
    const c10::optional<double> scales_w) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  const auto input_sizes = input.sizes();

  TORCH_CHECK(
      (4 == input_sizes.size()) && (2 == output_sizes.size()),
      "Invalid input!");

  vTensor v_output{
    context,
    {
      input_sizes[Layout::Activation4D::batch],
      input_sizes[Layout::Activation4D::channels],
      output_sizes[Layout::Parameter::height],
      output_sizes[Layout::Parameter::width],
    },
    input.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_input.has_image()) {
      const struct {
        float scale_x, scale_y;
      } block {
        compute_scales_value<float>(
            scales_w,
            input_sizes[Layout::Activation4D::width],
            output_sizes[Layout::Parameter::width]),
        compute_scales_value<float>(
            scales_h,
            input_sizes[Layout::Activation4D::height],
            output_sizes[Layout::Parameter::height]),
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(upsample_nearest2d),
          v_output.extents(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(command_buffer, vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_input.image(command_buffer),
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
  m.impl("upsample_nearest2d", TORCH_FN(upsample_nearest2d));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
