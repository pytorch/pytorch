#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor addmm(
    const Tensor& self_arg,
    const Tensor& mat1_arg,
    const Tensor& mat2_arg,
    const Scalar beta,
    const Scalar alpha) {
  api::Context* const context = api::context();
  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  const Tensor mat1 = mat1_arg.is_vulkan() ? mat1_arg : mat1_arg.vulkan();
  const vTensor& v_mat1 = convert(mat1);

  const Tensor mat2 = mat2_arg.is_vulkan() ? mat2_arg : mat2_arg.vulkan();
  const vTensor& v_mat2 = convert(mat2);

  vTensor v_output{
    context,
    {mat1.sizes()[0], mat2.sizes()[1]},
    self.options()
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_self.has_image()) {
      const struct {
        uint32_t width, height, channels;
        float beta, alpha;
        uint32_t k;
      } block {
        mat2_arg.sizes()[1],
        mat1_arg.sizes()[0],
        1u,
        beta.to<float>(),
        alpha.to<float>(),
        mat1_arg.sizes()[1],
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          },
          VK_KERNEL(addmm),
          v_output.extents(),
          v_output.image(command_buffer, vTensor::Access::Write),
          v_mat1.image(command_buffer),
          v_mat2.image(command_buffer),
          context->resource().pool.uniform(block).object,
          v_self.image(command_buffer));
    } else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

  return convert(v_output);
}

Tensor mm(const Tensor& self_arg, const Tensor& mat2_arg) {
  api::Context* const context = api::context();
  const Tensor mat1 = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_mat1 = convert(mat1);

  const Tensor mat2 = mat2_arg.is_vulkan() ? mat2_arg : mat2_arg.vulkan();
  const vTensor& v_mat2 = convert(mat2);

  vTensor v_output{
    context,
    {mat1.sizes()[0], mat2.sizes()[1]},
    mat1.options()
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_mat1.has_image() && v_mat2.has_image()) {
      const struct {
        uint32_t width, height, channels, k;
      } block {
        mat2.sizes()[1],
        mat1.sizes()[0],
        1u,
        mat1.sizes()[1],
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(mm),
          v_output.extents(),
          v_output.image(command_buffer, vTensor::Access::Write),
          v_mat1.image(command_buffer),
          v_mat2.image(command_buffer),
          context->resource().pool.uniform(block).object);
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
  m.impl("addmm", TORCH_FN(addmm));
  m.impl("mm", TORCH_FN(mm));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
