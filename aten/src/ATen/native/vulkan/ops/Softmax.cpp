#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor softmax_internal(
    const at::Tensor& input_arg,
    const int64_t dim,
    const bool half_to_float,
    const api::Shader::Descriptor& shader_descriptor,
    const std::string& op_name) {
  TORCH_CHECK(
      input_arg.dim() == 4,
      "Vulkan softmax expects 4-dimensional input!");

  TORCH_CHECK(
      dim == 1,
      "Vulkan softmax expects dim == 1 (channel)");

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  TORCH_CHECK(
      v_input_sizes[Layout::Activation4D::batch] == 1,
      "Vulkan softmax expects batch dim == 1");

  api::Context* const context = api::context();

  c10::SmallVector<int64_t, 4u> output_sizes{
    v_input_sizes[Layout::Activation4D::batch],
    v_input_sizes[Layout::Activation4D::channels],
    v_input_sizes[Layout::Activation4D::height],
    v_input_sizes[Layout::Activation4D::width],
  };

  vTensor v_output{
    context,
    output_sizes,
    v_input.options(),
  };

  const api::Shader::WorkGroup global_work_group_size = {
    safe_downcast<uint32_t>(v_input_sizes[Layout::Activation4D::width]),
    safe_downcast<uint32_t>(v_input_sizes[Layout::Activation4D::height]),
    1,
  };
  const api::Shader::WorkGroup local_work_group_size = {8, 8, 1};

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY(v_input.has_image()) {
      const struct Block final {
        uvec3 iextents;
        int last_texel_end_offset;
      } block {
        v_input.extents(),
        safe_downcast<int32_t>(
            (v_input_sizes[Layout::Activation4D::channels] - 1) % 4
        )
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          shader_descriptor,
          global_work_group_size,
          local_work_group_size,
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_input.image(
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

Tensor softmax(
    const at::Tensor& input_arg,
    const int64_t dim,
    const bool half_to_float) {
  return softmax_internal(input_arg, dim, half_to_float, VK_KERNEL(softmax), "_softmax");
}

Tensor log_softmax(
    const at::Tensor& input_arg,
    const int64_t dim,
    const bool half_to_float) {
  return softmax_internal(input_arg, dim, half_to_float, VK_KERNEL(log_softmax), "_log_softmax");
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("_softmax", TORCH_FN(softmax));
  m.impl("_log_softmax", TORCH_FN(log_softmax));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
