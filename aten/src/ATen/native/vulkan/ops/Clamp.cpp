#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

Tensor clamp(
    const Tensor& self_arg,
    const c10::optional<Scalar> min_value,
    const c10::optional<Scalar> max_value) {
  TORCH_CHECK(
      min_value || max_value,
      "At least one of 'min' or 'max' must not be None");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
    context,
    self.sizes(),
    self.options(),
  };

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_output.has_image() && v_self.has_image()) {
      const struct {
        float min_value;
        float max_value;
      } block {
        min_value ? min_value->to<float>() : -std::numeric_limits<float>::infinity(),
        max_value ? max_value->to<float>() : std::numeric_limits<float>::infinity(),
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(clamp),
          v_output.extents(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(command_buffer, vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_self.image(command_buffer),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
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

Tensor& clamp_(
    Tensor& self_arg,
    const c10::optional<Scalar> min_value,
    const c10::optional<Scalar> max_value) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      min_value || max_value,
      "At least one of 'min' or 'max' must not be None");

  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place clamp is only supported on Vulkan tensors.");

  vTensor& v_self = convert(self_arg);

  api::Command::Buffer command_buffer = context->command().pool.allocate();
  command_buffer.begin();
  {
    if (v_self.has_image()) {
      const struct {
        float min_value;
        float max_value;
      } block {
        min_value ? min_value->to<float>() : -std::numeric_limits<float>::infinity(),
        max_value ? max_value->to<float>() : std::numeric_limits<float>::infinity(),
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(clamp_),
          v_self.extents(),
          // Read-Write access triggers an async synchronization if necessory
          // and inserts appropriate barriers if hazards are detected.
          v_self.image(command_buffer, vTensor::Access::Read | vTensor::Access::Write),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_buffer.end();
  command_buffer.submit(context->gpu().queue);

  return self_arg;
}

Tensor hardtanh(
    const Tensor& self,
    const Scalar min,
    const Scalar max) {
  return ops::clamp(self, min, max);
}

Tensor& hardtanh_(
    Tensor& self,
    const Scalar min,
    const Scalar max) {
  return ops::clamp_(self, min, max);
}

Tensor relu(const Tensor& self) {
  return ops::clamp(self, 0, c10::nullopt);
}

Tensor& relu_(Tensor& self) {
  return ops::clamp_(self, 0, c10::nullopt);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("clamp", TORCH_FN(clamp));
  m.impl("clamp_", TORCH_FN(clamp_));
  m.impl_UNBOXED("hardtanh", hardtanh);
  m.impl_UNBOXED("hardtanh_", hardtanh_);
  m.impl_UNBOXED("relu", relu);
  m.impl_UNBOXED("relu_", relu_);
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
