#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor _clamp(
    const Tensor& self_arg,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max,
    const api::Shader::Descriptor& shader_descriptor,
    const std::string& op_name) {
  TORCH_CHECK(
      min || max,
      "At least one of 'min' or 'max' must not be None");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
    context,
    v_self.sizes(),
    v_self.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY(v_output.has_image() && v_self.has_image()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        vec2 clamp;
      } block {
        v_output.extents(),
        0u,
        {
          min ? min->to<float>() : -std::numeric_limits<float>::infinity(),
          max ? max->to<float>() : std::numeric_limits<float>::infinity(),
        },
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          shader_descriptor,
          v_output.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_self.image(
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

Tensor clamp(
    const Tensor& self_arg,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max) {
  return _clamp(self_arg, min, max, VK_KERNEL(clamp), "aten::clamp");
}

Tensor& _clamp_(
    Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max,
    const api::Shader::Descriptor& shader_descriptor,
    const std::string& op_name) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      min || max,
      "At least one of 'min' or 'max' must not be None");

  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place clamp is only supported on Vulkan tensors.");

  vTensor& v_self = convert(self);

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY(v_self.has_image()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        vec2 clamp;
      } block {
        v_self.extents(),
        0u,
        {
          min ? min->to<float>() : -std::numeric_limits<float>::infinity(),
          max ? max->to<float>() : std::numeric_limits<float>::infinity(),
        },
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          shader_descriptor,
          v_self.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Read-Write access triggers an async synchronization if necessory
          // and inserts appropriate barriers if hazards are detected.
          v_self.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Read | vTensor::Access::Write),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return self;
}

Tensor threshold(
    const Tensor& self,
    const Scalar& threshold,
    const Scalar& value) {
  return _clamp(self, threshold, value, VK_KERNEL(threshold), "aten::threshold");
}

Tensor& clamp_(
    Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max) {
  return _clamp_(self, min, max, VK_KERNEL(clamp_), "aten::clamp_");
}

Tensor activation(
    const Tensor& self_arg,
    const api::Shader::Descriptor& shader_descriptor,
    const std::string& op_name) {
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
    context,
    v_self.sizes(),
    v_self.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY(v_output.has_image() && v_self.has_image()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
      } block {
        v_output.extents(),
        0u,
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          shader_descriptor,
          v_output.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_self.image(
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

Tensor& activation_(
    Tensor& self,
    const api::Shader::Descriptor& shader_descriptor,
    const std::string& op_name) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  vTensor& v_self = convert(self);

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY(v_self.has_image()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
      } block {
        v_self.extents(),
        0u,
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          shader_descriptor,
          v_self.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Read-Write access triggers an async synchronization if necessory
          // and inserts appropriate barriers if hazards are detected.
          v_self.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Read | vTensor::Access::Write),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return self;
}

Tensor hardtanh(
    const Tensor& self,
    const Scalar& min,
    const Scalar& max) {
  return ops::_clamp(self, min, max, VK_KERNEL(clamp), "aten::hardtanh");
}

Tensor& hardtanh_(
    Tensor& self,
    const Scalar& min,
    const Scalar& max) {
  return ops::_clamp_(self, min, max, VK_KERNEL(clamp_), "aten::hardtanh_");
}

Tensor relu(const Tensor& self) {
  return ops::_clamp(self, 0, c10::nullopt, VK_KERNEL(clamp), "aten::relu");
}

Tensor& relu_(Tensor& self) {
  return ops::_clamp_(self, 0, c10::nullopt, VK_KERNEL(clamp_), "aten::relu_");
}

Tensor hardswish(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(hardswish), "aten::hardswish");
}

Tensor& hardswish_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(hardswish_), "aten::hardswish_");
}

Tensor hardsigmoid(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(hardsigmoid), "aten::hardsigmoid");
}

Tensor& hardsigmoid_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(hardsigmoid_), "aten::hardsigmoid_");
}

Tensor activation_scalar(
    const Tensor& self_arg,
    const Scalar& scalar_arg,
    const api::Shader::Descriptor& shader_descriptor,
    const std::string& op_name) {
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  vTensor v_output{
    context,
    v_self.sizes(),
    v_self.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY(v_output.has_image() && v_self.has_image()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scalar_value;
      } block {
        v_output.extents(),
        0u,
        scalar_arg.to<float>(),
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          shader_descriptor,
          v_output.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_self.image(
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

Tensor& activation_scalar_(
    Tensor& self,
    const Scalar& scalar_arg,
    const api::Shader::Descriptor& shader_descriptor,
    const std::string& op_name) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  vTensor& v_self = convert(self);

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY(v_self.has_image()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scalar_value;
      } block {
        v_self.extents(),
        0u,
        scalar_arg.to<float>(),
      };

      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          shader_descriptor,
          v_self.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Read-Write access triggers an async synchronization if necessory
          // and inserts appropriate barriers if hazards are detected.
          v_self.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Read | vTensor::Access::Write),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    }
    else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return self;
}

Tensor hardshrink(
    const Tensor& self_arg,
    const Scalar& lambd) {
  return ops::activation_scalar(self_arg, lambd, VK_KERNEL(hardshrink), "aten::hardshrink");
}

Tensor& hardshrink_(
    Tensor& self,
    const Scalar& lambd) {
  return ops::activation_scalar_(self, lambd, VK_KERNEL(hardshrink_), "aten::hardshrink_");
}

Tensor leaky_relu(
    const Tensor& self_arg,
    const Scalar& negative_slope) {
  return ops::activation_scalar(self_arg, negative_slope, VK_KERNEL(leaky_relu), "aten::leaky_relu");
}

Tensor& leaky_relu_(
    Tensor& self,
    const Scalar& negative_slope) {
  return ops::activation_scalar_(self, negative_slope, VK_KERNEL(leaky_relu_), "aten::leaky_relu_");
}

Tensor sigmoid(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(sigmoid), "aten::sigmoid");
}

Tensor& sigmoid_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(sigmoid_), "aten::sigmoid_");
}

Tensor tanh(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(tanh), "aten::tanh");
}

Tensor& tanh_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(tanh_), "aten::tanh_");
}


#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::clamp"), TORCH_FN(clamp));
  m.impl(TORCH_SELECTIVE_NAME("aten::clamp_"), TORCH_FN(clamp_));
  m.impl(TORCH_SELECTIVE_NAME("aten::hardsigmoid"), hardsigmoid);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardsigmoid_"), hardsigmoid_);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardshrink"), hardshrink);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardshrink_"), hardshrink_);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardswish"), hardswish);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardswish_"), hardswish_);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardtanh"), hardtanh);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardtanh_"), hardtanh_);
  m.impl(TORCH_SELECTIVE_NAME("aten::leaky_relu"), leaky_relu);
  m.impl(TORCH_SELECTIVE_NAME("aten::leaky_relu_"), leaky_relu_);
  m.impl(TORCH_SELECTIVE_NAME("aten::sigmoid"), sigmoid);
  m.impl(TORCH_SELECTIVE_NAME("aten::sigmoid_"), sigmoid_);
  m.impl(TORCH_SELECTIVE_NAME("aten::tanh"), tanh);
  m.impl(TORCH_SELECTIVE_NAME("aten::tanh_"), tanh_);
  m.impl(TORCH_SELECTIVE_NAME("aten::relu"), relu);
  m.impl(TORCH_SELECTIVE_NAME("aten::relu_"), relu_);
  m.impl(TORCH_SELECTIVE_NAME("aten::threshold"), threshold);
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
