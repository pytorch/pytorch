#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor adaptive_avg_pool2d(const at::Tensor& input_arg, IntArrayRef output_size) {
  TORCH_INTERNAL_ASSERT(
      input_arg.dim() == 4,
      "vulkan_adaptive_avg_pool2d expects 4-dimensional input");

  api::Context* const context = api::context();
  const vTensor& v_input = convert(input_arg);
  vTensor v_output{
    context,
    {input_arg.sizes()[0], input_arg.sizes()[1], output_size[0], output_size[1]},
    input_arg.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_input.has_image()) {
      const struct {
        uint32_t input_width, input_height, output_width, output_height;
      } block {
        input_arg.sizes()[3],
        input_arg.sizes()[2],
        output_size[0],
        output_size[1],
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
  m.impl("_adaptive_avg_pool2d", TORCH_FN(adaptive_avg_pool2d));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
