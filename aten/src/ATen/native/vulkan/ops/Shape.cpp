#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor view(
    const Tensor& self_arg,
    IntArrayRef shape) {
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
    context,
    shape,
    self.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    command_buffer.copy(
        // Read-only access is implied on const tensors and triggers an async
        // synchronization if necessary.
        v_self.buffer(command_buffer),
        // Write-only access bypasses synchronization but inserts appropriate
        // barriers if necessary.
        v_output.buffer(command_buffer, vTensor::Access::Write));
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("view", TORCH_FN(view));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
