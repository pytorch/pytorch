#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor view_internal(
    const Tensor& self_arg,
    const IntArrayRef shape,
    const std::string& op_name) {
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
    context,
    shape,
    self.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    command_buffer.copy(
        // Read-only access is implied on const tensors and triggers an async
        // synchronization if necessary.
        v_self.buffer(
            command_buffer,
            vTensor::Stage::Transfer),
        // Write-only access bypasses synchronization but inserts appropriate
        // barriers if necessary.
        v_output.buffer(
            command_buffer,
            vTensor::Stage::Transfer,
            vTensor::Access::Write));
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return convert(v_output);
}

inline Tensor view(
    const Tensor& self_arg,
    const IntArrayRef shape) {
  return view_internal(self_arg, shape, "aten::view");
}

Tensor _reshape_alias(
    const Tensor& self_arg,
    const IntArrayRef shape,
    const IntArrayRef strides) {
  return view_internal(self_arg, shape, "aten::_reshape_alias");
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::view"), TORCH_FN(view));
  m.impl(TORCH_SELECTIVE_NAME("aten::_reshape_alias"), TORCH_FN(_reshape_alias));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
