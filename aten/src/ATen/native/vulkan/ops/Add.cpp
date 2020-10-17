#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/glsl.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor add(
    const Tensor& self,
    const Tensor& other,
    const Scalar alpha) {
  api::Context* const context = api::context();

  const vTensor& v_self = convert(self.is_vulkan() ? self : self.vulkan());
  const vTensor& v_other = convert(other.is_vulkan() ? other : self.vulkan());
  vTensor v_output(context, self.sizes().vec(), self.options());

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();

  api::Descriptor::Set descriptor_set = context->load(
      command_buffer,
      {
        {
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        },
      },
      {
        add_glsl,
      },
      {
        8, 8, 1,
      });

  descriptor_set.
      bind(0u, v_output.image(command_buffer, vTensor::Access::Write)).
      bind(1u, v_self.image(command_buffer)).
      bind(2u, v_other.image(command_buffer));

  context->dispatch(
      command_buffer,
      descriptor_set,
      {
      });

  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("add.Tensor", TORCH_FN(at::native::vulkan::ops::add));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
