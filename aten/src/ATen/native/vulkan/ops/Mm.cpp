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

  const auto self_sizes = self.sizes();
  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();

  if (self_sizes.size() >= 2) {
    TORCH_CHECK(
        (mat1_sizes[Layout::Parameter::width] ==
            mat2_sizes[Layout::Parameter::height]) &&
        (self_sizes[Layout::Parameter::height] ==
            mat1_sizes[Layout::Parameter::height]) &&
        (self_sizes[Layout::Parameter::width] ==
            mat2_sizes[Layout::Parameter::width]),
        "Incompatible matrix dimensions!");
  }
  else {
    TORCH_CHECK(
        (mat1_sizes[Layout::Parameter::width] ==
            mat2_sizes[Layout::Parameter::height]) &&
        ((self_sizes[Layout::Parameter::height] ==
            mat1_sizes[Layout::Parameter::height]) ||
         (self_sizes[Layout::Parameter::height] ==
            mat2_sizes[Layout::Parameter::width])),
        "Incompatible matrix dimensions!");
  }

  vTensor v_output{
    context,
    {
      mat1_sizes[Layout::Parameter::height],
      mat2_sizes[Layout::Parameter::width],
    },
    self.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_self.has_image()) {
      const struct {
        float beta, alpha;
      } block {
        alpha.to<float>(),
        beta.to<float>(),
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(addmm),
          v_output.extents(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(command_buffer, vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_mat1.image(command_buffer),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_mat2.image(command_buffer),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_self.image(command_buffer),
          context->resource().pool.uniform(block).object);
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

  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();

  TORCH_CHECK(
      mat1_sizes[Layout::Parameter::width] ==
          mat2_sizes[Layout::Parameter::height],
      "Incompatible matrix dimensions!");

  vTensor v_output{
    context,
    {
      mat1_sizes[Layout::Parameter::height],
      mat2_sizes[Layout::Parameter::width],
    },
    mat1.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_mat1.has_image() && v_mat2.has_image()) {
      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          },
          VK_KERNEL(mm),
          v_output.extents(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(command_buffer, vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_mat1.image(command_buffer),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_mat2.image(command_buffer));
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
