#include <ATen/native/UpSample.h>
#include <ATen/native/vulkan/ops/Common.h>
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

  vTensor v_output{
    context,
    {input_arg.sizes()[0], input_arg.sizes()[1], output_sizes[0], output_sizes[1]},
    input.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    const float scale_x = compute_scales_value<float>(scales_w, input_arg.sizes()[3], output_sizes[1]);
    const float scale_y = compute_scales_value<float>(scales_h, input_arg.sizes()[2], output_sizes[0]);
    if (v_input.has_image()) {
      const struct {
        uint32_t input_width, input_height, output_width, output_height;
        float scale_x, scale_y;
      } block {
        input_arg.sizes()[3],
        input_arg.sizes()[2],
        output_sizes[1],
        output_sizes[0],
        scale_x,
        scale_y
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
          v_output.image(command_buffer, vTensor::Access::Write),
          v_input.image(command_buffer),
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
