#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

int64_t normalize_dim(int64_t d, int64_t n) {
  return (d % n + n) % n;
}

Tensor mean(
    const at::Tensor& input_arg,
    const IntArrayRef dim,
    const bool keepdim,
    const optional<ScalarType> dtype) {
  TORCH_INTERNAL_ASSERT(
      input_arg.dim() == 4, "vulkan_mean expects 4-dimensional input");
  static const std::unordered_set<int64_t> expected_dims_set({2, 3});
  std::unordered_set<int64_t> dims_set;
  for (const auto& d : dim) {
    dims_set.insert(normalize_dim(d, 4));
  }
  TORCH_INTERNAL_ASSERT(
      dims_set == expected_dims_set,
      "vulkan_mean currently only supported for image-wide reduction");

  std::vector<int64_t> output_dims{input_arg.sizes()[0], input_arg.sizes()[1]};
  if (keepdim) {
    output_dims.push_back(1);
    output_dims.push_back(1);
  }

  api::Context* const context = api::context();
  const vTensor& v_input = convert(input_arg);
  vTensor v_output{
      context,
      output_dims,
      input_arg.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_input.has_image()) {
      const struct {
        uint32_t input_width, input_height;
      } block{
          input_arg.sizes()[3],
          input_arg.sizes()[2],
      };

      if (keepdim) {
        context->dispatch(
            command_buffer,
            {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            },
            VK_KERNEL(mean),
            v_output.extents(),
            v_output.image(command_buffer, vTensor::Access::Write),
            v_input.image(command_buffer));
      } else {
        context->dispatch(
            command_buffer,
            {
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            },
            VK_KERNEL(mean2d),
            v_output.extents(),
            v_output.image(command_buffer, vTensor::Access::Write),
            v_input.image(command_buffer),
            context->resource().pool.uniform(block).object);
      }
    } else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("mean.dim", TORCH_FN(mean));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
