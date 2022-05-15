#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

void check_inputs(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(
      channels_size(input1) == channels_size(input2),
      "Vulkan binary elementwise ops require channel dimension to be equal!");
  if (batch_size(input1) != batch_size(input2)) {
    TORCH_CHECK(
        channels_size(input1) % 4 == 0,
        "Vulkan binary elementwise ops require channel to be a multiple of 4 to broadcast along batch dimension!")
  }

  const uint32_t input1_h = height_size(input1);
  const uint32_t input1_w = width_size(input1);
  const uint32_t input2_h = height_size(input2);
  const uint32_t input2_w = width_size(input2);

  const std::string broadcast_error_msg =
      "Incompatible input dimensions for broadcasting for Vulkan binary elementwise op!";
  if (input1_h != input2_h) {
    if (input1_h > input2_h) {
      TORCH_CHECK(input2_h == 1, broadcast_error_msg);
      TORCH_CHECK(input2_w == input1_w || input2_w == 1, broadcast_error_msg);
    } else if (input2_h > input1_h) {
      TORCH_CHECK(input1_h == 1, broadcast_error_msg);
      TORCH_CHECK(input1_w == input2_w || input1_w == 1, broadcast_error_msg);
    }
  } else if (input1_w != input2_w) {
    if (input1_w > input2_w) {
      TORCH_CHECK(input2_w == 1, broadcast_error_msg);
    } else if (input2_w > input1_w) {
      TORCH_CHECK(input1_h == 1, broadcast_error_msg);
    }
  }
}

bool broadcast_first_input(const vTensor& input1, const vTensor& input2) {
  return (
      (input2.extents().data[1u] > 1 && input1.extents().data[1u] == 1) ||
      (input2.extents().data[2u] > 1 && input1.extents().data[2u] == 1) ||
      input2.extents().data[0u] > input1.extents().data[0u]);
}

Tensor arithmetic_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const c10::optional<Scalar>& alpha_arg,
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

    if C10_LIKELY (v_output.has_image() && v_self.has_image()) {
      const float other_val = alpha_arg
          ? other.to<float>() * alpha_arg->to<float>()
          : other.to<float>();
      const struct Block final {
        uvec3 extents;
        float other;
      } block{
          v_self.extents(),
          other_val,
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
          adaptive_work_group_size(v_output.extents()),
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

Tensor& arithmetic_scalar_(
    Tensor& self,
    const Scalar& other,
    const c10::optional<Scalar>& alpha_arg,
    const api::Shader::Descriptor& shader_descriptor,
    const std::string& op_name) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place add is only supported on Vulkan tensors.");

  vTensor& v_self = convert(self);

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY (v_self.has_image()) {
      const float other_val = alpha_arg
          ? other.to<float>() * alpha_arg->to<float>()
          : other.to<float>();
      const struct Block final {
        uvec3 extents;
        float other;
      } block{
          v_self.extents(),
          other_val,
      };

      context->dispatch(
          command_buffer,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          shader_descriptor,
          v_self.extents(),
          adaptive_work_group_size(v_self.extents()),
          // Read-Write access triggers an async synchronization if necessory
          // and inserts appropriate barriers if hazards are detected.
          v_self.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Read | vTensor::Access::Write),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    } else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return self;
}

Tensor arithmetic_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const c10::optional<Scalar>& alpha_arg,
    const api::Shader::Descriptor& shader_descriptor,
    const std::string& op_name) {
  check_inputs(self_arg, other_arg);
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  const vTensor& v_other = convert(other);

  vTensor v_output{
      context,
      broadcast_first_input(v_self, v_other) ? v_other.sizes() : v_self.sizes(),
      v_self.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY (v_self.has_image() && v_other.has_image()) {
      const float alpha = alpha_arg ? alpha_arg->to<float>() : 1.0;
      const struct Block final {
        uvec3 extents;
        uint32_t fill_0;
        uvec3 input1_extents;
        uint32_t fill_1;
        uvec3 input2_extents;
        float alpha;
      } block{
          v_output.extents(),
          0u,
          v_self.extents(),
          0u,
          v_other.extents(),
          alpha,
      };

      context->dispatch(
          command_buffer,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          shader_descriptor,
          v_output.extents(),
          adaptive_work_group_size(v_output.extents()),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer, vTensor::Stage::Compute, vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_self.image(command_buffer, vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_other.image(command_buffer, vTensor::Stage::Compute),
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

Tensor& arithmetic_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const c10::optional<Scalar>& alpha_arg,
    const api::Shader::Descriptor& shader_descriptor,
    const std::string& op_name) {
  check_inputs(self, other_arg);
  api::Context* const context = api::context();

  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place add is only supported on Vulkan tensors.");

  vTensor& v_self = convert(self);

  const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  const vTensor& v_other = convert(other);

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY (
        v_self.has_image() && v_other.has_image() && !self.is_same(other)) {
      const float alpha = alpha_arg ? alpha_arg->to<float>() : 1.0;
      const struct Block final {
        uvec3 extents;
        uint32_t fill_0;
        uvec3 input_extents;
        float alpha;
      } block{
          v_self.extents(),
          0u,
          v_other.extents(),
          alpha,
      };

      context->dispatch(
          command_buffer,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          shader_descriptor,
          v_self.extents(),
          adaptive_work_group_size(v_self.extents()),
          // Read-Write access triggers an async synchronization if necessory
          // and inserts appropriate barriers if hazards are detected.
          v_self.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Read | vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_other.image(command_buffer, vTensor::Stage::Compute),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    } else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return self;
}

Tensor add_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  return arithmetic_scalar(
      self_arg, other, c10::optional<Scalar>(alpha), VK_KERNEL(add_scalar), "aten::add.Scalar");
}

Tensor& add_scalar_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return arithmetic_scalar_(
      self, other, c10::optional<Scalar>(alpha), VK_KERNEL(add_scalar_), "aten::add_.Scalar");
}

Tensor add_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  if (other_arg.sizes().size() == 0) {
    return arithmetic_scalar(
        self_arg,
        other_arg.item<float>(),
        c10::optional<Scalar>(alpha.to<float>()),
        VK_KERNEL(add_scalar),
        "aten::add.Tensor");
  }
  return arithmetic_tensor(
      self_arg, other_arg, c10::optional<Scalar>(alpha), VK_KERNEL(add), "aten::add.Tensor");
}

Tensor& add_tensor_(Tensor& self, const Tensor& other_arg, const Scalar& alpha) {
  return arithmetic_tensor_(
      self, other_arg, c10::optional<Scalar>(alpha), VK_KERNEL(add_), "aten::add_.Tensor");
}

Tensor sub_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  return arithmetic_scalar(
      self_arg,
      other,
      c10::optional<Scalar>(-1 * alpha.to<float>()),
      VK_KERNEL(add_scalar),
      "aten::sub.Scalar");
}

Tensor& sub_scalar_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return arithmetic_scalar_(
      self,
      other,
      c10::optional<Scalar>(-1 * alpha.to<float>()),
      VK_KERNEL(add_scalar_),
      "aten::sub_.Scalar");
}

Tensor sub_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  if (other_arg.sizes().size() == 0) {
    return arithmetic_scalar(
        self_arg,
        other_arg.item<float>(),
        c10::optional<Scalar>(-1 * alpha.to<float>()),
        VK_KERNEL(add_scalar),
        "aten::sub.Tensor");
  }
  return arithmetic_tensor(
      self_arg, other_arg, c10::optional<Scalar>(alpha), VK_KERNEL(sub), "aten::sub.Tensor");
}

Tensor& sub_tensor_(Tensor& self, const Tensor& other_arg, const Scalar& alpha) {
  return arithmetic_tensor_(
      self, other_arg, c10::optional<Scalar>(alpha), VK_KERNEL(sub_), "aten::sub_.Tensor");
}

Tensor mul_scalar(const Tensor& self_arg, const Scalar& other) {
  return arithmetic_scalar(
      self_arg, other, c10::optional<Scalar>(), VK_KERNEL(mul_scalar), "aten::mul.Scalar");
}

Tensor& mul_scalar_(Tensor& self, const Scalar& other) {
  return arithmetic_scalar_(
      self, other, c10::optional<Scalar>(), VK_KERNEL(mul_scalar_), "aten::mul_.Scalar");
}

Tensor mul_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  if (other_arg.sizes().size() == 0) {
    return arithmetic_scalar(
        self_arg,
        other_arg.item<float>(),
        c10::optional<Scalar>(),
        VK_KERNEL(mul_scalar),
        "aten::mul.Tensor");
  }
  return arithmetic_tensor(
      self_arg, other_arg, c10::optional<Scalar>(), VK_KERNEL(mul), "aten::mul.Tensor");
}

Tensor& mul_tensor_(Tensor& self, const Tensor& other_arg) {
  return arithmetic_tensor_(
      self, other_arg, c10::optional<Scalar>(), VK_KERNEL(mul_), "aten::mul_.Tensor");
}

Tensor div_scalar(const Tensor& self_arg, const Scalar& other) {
  return arithmetic_scalar(
      self_arg,
      1.0 / other.to<float>(),
      c10::optional<Scalar>(),
      VK_KERNEL(mul_scalar),
      "aten::div.Scalar");
}

Tensor& div_scalar_(Tensor& self, const Scalar& other) {
  return arithmetic_scalar_(
      self,
      1.0 / other.to<float>(),
      c10::optional<Scalar>(),
      VK_KERNEL(mul_scalar_),
      "aten::div_.Scalar");
}

Tensor div_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  if (other_arg.sizes().size() == 0) {
    return arithmetic_scalar(
        self_arg,
        1.0 / other_arg.item<float>(),
        c10::optional<Scalar>(),
        VK_KERNEL(mul_scalar),
        "aten::div.Tensor");
  }
  return arithmetic_tensor(
      self_arg, other_arg, c10::optional<Scalar>(), VK_KERNEL(div), "aten::div.Tensor");
}

Tensor& div_tensor_(Tensor& self, const Tensor& other_arg) {
  return arithmetic_tensor_(
      self, other_arg, c10::optional<Scalar>(), VK_KERNEL(div_), "aten::div_.Tensor");
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Scalar"), TORCH_FN(add_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Scalar"), TORCH_FN(add_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Tensor"), TORCH_FN(add_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Tensor"), TORCH_FN(add_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Scalar"), TORCH_FN(sub_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Scalar"), TORCH_FN(sub_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Tensor"), TORCH_FN(sub_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Tensor"), TORCH_FN(sub_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Scalar"), TORCH_FN(mul_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Scalar"), TORCH_FN(mul_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Tensor"), TORCH_FN(mul_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Tensor"), TORCH_FN(mul_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Scalar"), TORCH_FN(div_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Scalar"), TORCH_FN(div_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Tensor"), TORCH_FN(div_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Tensor"), TORCH_FN(div_tensor_));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
