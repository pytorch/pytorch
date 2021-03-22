#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor reflection_pad2d(const Tensor& self_arg, IntArrayRef padding) {
  const int pad_dim = padding.size();
  const IntArrayRef input_size = self_arg.sizes();
  const int input_dim = input_size.size();

  TORCH_CHECK(
      pad_dim == 1 || pad_dim == 4,
      "Padding sizes must be a 1-tuple or 4-tuple!");
  TORCH_CHECK(input_dim >= 2, "Input tensor must have dim >= 2!");

  api::Context* const context = api::context();

  int pad_left = padding[0];
  int pad_right = padding[0];
  int pad_top = padding[0];
  int pad_bottom = padding[0];
  if (pad_dim == 4) {
    pad_right = padding[1];
    pad_top = padding[2];
    pad_bottom = padding[3];
  }

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  c10::SmallVector<int64_t, 4> output_size(input_dim);
  for (size_t d = 0; d < input_dim; ++d) {
    if (d == input_dim - 1) {
      output_size[d] = input_size[d] + pad_right + pad_left;
    } else if (d == input_dim - 2) {
      output_size[d] = input_size[d] + pad_top + pad_bottom;
    } else {
      output_size[d] = input_size[d];
    }
  }

  vTensor v_output{
      context,
      output_size,
      v_self.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    if C10_LIKELY (v_output.has_image() && v_self.has_image()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        uvec4 padding;
      } block{
          v_output.extents(),
          0u,
          {pad_left, pad_right, pad_top, pad_bottom},
      };

      context->dispatch(
          command_buffer,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(reflection_pad2d),
          v_output.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer, vTensor::Stage::Compute, vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_self.image(command_buffer, vTensor::Stage::Compute),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    } else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("reflection_pad2d", TORCH_FN(reflection_pad2d));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
