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

Tensor cumsum(
    const at::Tensor& input_arg,
    const int64_t dim,
    const c10::optional<ScalarType> dtype) {
  TORCH_CHECK(
    input_arg.dim() <= 4,
    "Vulkan cumsum expects input dimension <= 4!");

  TORCH_CHECK(
    batch_size(input_arg) == 1,
    "Vulkan cumsum expects batch size <= 1!");

  TORCH_CHECK(
    dim < 4,
    "Vulkan cumsum expects dim < 4!");

  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  vTensor v_output{
    context,
    input.sizes(),
    input.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), "aten::cumsum");

    if C10_LIKELY(v_input.has_image()) {
      const struct Block final {
        int32_t axis;
      } block {
        (3-safe_downcast<int32_t>(dim)),
      };

      if(dim<=1) {
          // TODO: dim<0, dim=0, dim=1(z axis)
          TORCH_CHECK(false, "Not implemented!");
      }

      api::UniformParamsBuffer params(context, block);

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(cumsum),
          v_input.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer,
              api::PipelineStage::Compute,
              api::MemoryAccessType::WRITE),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_input.image(
              command_buffer,
              api::PipelineStage::Compute),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          params.buffer().package());
    } else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::cumsum"), TORCH_FN(cumsum));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
