#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

void check_inputs_elementwise_op(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(
      channels_size(input1) == channels_size(input2),
      "Vulkan elementwise ops require channel dimension to be equal!");
  if (batch_size(input1) != batch_size(input2)) {
    TORCH_CHECK(
        channels_size(input1) % 4 == 0,
        "Vulkan elementwise ops require channel to be a multiple of 4 to broadcast along batch dimension!")
  }

  const uint32_t input1_h = height_size(input1);
  const uint32_t input1_w = width_size(input1);
  const uint32_t input2_h = height_size(input2);
  const uint32_t input2_w = width_size(input2);

  const std::string broadcast_error_msg =
      "Incompatible input dimensions for broadcasting for Vulkan elementwise op!";
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

Tensor _lerp_scalar(
    const Tensor& start_arg,
    const Tensor& end_arg,
    const Scalar& weight_arg,
    const std::string& op_name) {
  check_inputs_elementwise_op(start_arg, end_arg);
  api::Context* const context = api::context();

  const Tensor start = start_arg.is_vulkan() ? start_arg : start_arg.vulkan();
  const vTensor& v_start = convert(start);

  const Tensor end = end_arg.is_vulkan() ? end_arg : end_arg.vulkan();
  const vTensor& v_end = convert(end);

  vTensor v_output{
    context,
    v_start.sizes(),
    v_start.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY (v_start.has_image() && v_end.has_image()) {
      const float weight = weight_arg.to<float>();
      const struct Block final {
        uvec3 extents;
        uint32_t fill_0;
        uvec3 input1_extents;
        uint32_t fill_1;
        uvec3 input2_extents;
        float weight;
      } block{
          v_output.extents(),
          0u,
          v_start.extents(),
          0u,
          v_end.extents(),
          weight,
      };

      api::UniformParamsBuffer params(context, block);

      context->dispatch(
          command_buffer,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(lerp_scalar),
          v_output.extents(),
          adaptive_work_group_size(v_output.extents()),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer, vTensor::Stage::Compute, api::MemoryAccessType::WRITE),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_start.image(command_buffer, vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_end.image(command_buffer, vTensor::Stage::Compute),
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

Tensor& _lerp_scalar_(
    Tensor& self,
    const Tensor& end_arg,
    const Scalar& weight_arg,
    const std::string& op_name) {
  check_inputs_elementwise_op(self, end_arg);
  api::Context* const context = api::context();

  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place lerp is only supported on Vulkan tensors.");

  vTensor& v_self = convert(self);

  const Tensor end = end_arg.is_vulkan() ? end_arg : end_arg.vulkan();
  const vTensor& v_end = convert(end);

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY (
        v_self.has_image() && v_end.has_image() && !self.is_same(end)) {
      const float weight = weight_arg.to<float>();
      const struct Block final {
        uvec3 extents;
        uint32_t fill_0;
        uvec3 input_extents;
        float alpha;
      } block{
          v_self.extents(),
          0u,
          v_end.extents(),
          weight,
      };

      api::UniformParamsBuffer params(context, block);

      context->dispatch(
          command_buffer,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(lerp_scalar_),
          v_self.extents(),
          adaptive_work_group_size(v_self.extents()),
          // Read-Write access triggers an async synchronization if necessory
          // and inserts appropriate barriers if hazards are detected.
          v_self.image(
              command_buffer,
              vTensor::Stage::Compute,
              api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_end.image(command_buffer, vTensor::Stage::Compute),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          params.buffer().package());
    } else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return self;
}

Tensor _lerp_tensor(
    const Tensor& start_arg,
    const Tensor& end_arg,
    const Tensor& weight_arg,
    const std::string& op_name) {
  check_inputs_elementwise_op(start_arg, end_arg);
  check_inputs_elementwise_op(start_arg, weight_arg);
  api::Context* const context = api::context();

  const Tensor start = start_arg.is_vulkan() ? start_arg : start_arg.vulkan();
  const vTensor& v_start = convert(start);

  const Tensor end = end_arg.is_vulkan() ? end_arg : end_arg.vulkan();
  const vTensor& v_end = convert(end);

  const Tensor weight = weight_arg.is_vulkan() ? weight_arg : weight_arg.vulkan();
  const vTensor& v_weight = convert(weight);

  vTensor v_output{
    context,
    v_start.sizes(),
    v_start.options(),
  };

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY (v_start.has_image() && v_end.has_image() && v_weight.has_image()) {
      const struct Block final {
        uvec3 extents;
        uint32_t fill_0;
        uvec3 input1_extents;
        uint32_t fill_1;
        uvec3 input2_extents;
        uint32_t fill_2;
        uvec3 input3_extents;
        uint32_t fill_3;
      } block{
          v_output.extents(),
          0u,
          v_start.extents(),
          0u,
          v_end.extents(),
          0u,
          v_weight.extents(),
          0u,
      };

      api::UniformParamsBuffer params(context, block);

      context->dispatch(
          command_buffer,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(lerp),
          v_output.extents(),
          adaptive_work_group_size(v_output.extents()),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer, vTensor::Stage::Compute, api::MemoryAccessType::WRITE),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_start.image(command_buffer, vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_end.image(command_buffer, vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_weight.image(command_buffer, vTensor::Stage::Compute),
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

Tensor& _lerp_tensor_(
    Tensor& self,
    const Tensor& end_arg,
    const Tensor& weight_arg,
    const std::string& op_name) {
  check_inputs_elementwise_op(self, end_arg);
  check_inputs_elementwise_op(self, weight_arg);
  api::Context* const context = api::context();

  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place lerp is only supported on Vulkan tensors.");

  vTensor& v_self = convert(self);

  const Tensor end = end_arg.is_vulkan() ? end_arg : end_arg.vulkan();
  const vTensor& v_end = convert(end);

  const Tensor weight = weight_arg.is_vulkan() ? weight_arg : weight_arg.vulkan();
  const vTensor& v_weight = convert(weight);

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    api::OpProfiler profiler(command_buffer, context->querypool(), op_name);

    if C10_LIKELY (
        v_self.has_image() && v_end.has_image() && v_weight.has_image() && !self.is_same(end)) {
      const struct Block final {
        uvec3 extents;
        uint32_t fill_0;
        uvec3 input1_extents;
        uint32_t fill_1;
        uvec3 input2_extents;
        uint32_t fill_2;
      } block{
          v_self.extents(),
          0u,
          v_end.extents(),
          0u,
          v_weight.extents(),
          0u,
      };

      api::UniformParamsBuffer params(context, block);

      context->dispatch(
          command_buffer,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(lerp_),
          v_self.extents(),
          adaptive_work_group_size(v_self.extents()),
          // Read-Write access triggers an async synchronization if necessory
          // and inserts appropriate barriers if hazards are detected.
          v_self.image(
              command_buffer,
              vTensor::Stage::Compute,
              api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_end.image(command_buffer, vTensor::Stage::Compute),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_weight.image(command_buffer, vTensor::Stage::Compute),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          params.buffer().package());
    } else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);

  return self;
}

Tensor lerp_scalar(const Tensor& start, const Tensor& end, const Scalar& weight) {
  return _lerp_scalar(
      start, end, weight, "aten::lerp.Scalar");
}

Tensor& lerp_scalar_(Tensor& self, const Tensor& end, const Scalar& weight) {
  return _lerp_scalar_(
      self, end, weight, "aten::lerp_.Scalar");
}

Tensor lerp_tensor(const Tensor& start, const Tensor& end, const Tensor& weight) {
  if (weight.sizes().size() == 0) {
    return _lerp_scalar(
        start, end, weight.item<float>(), "aten::lerp.Tensor");
  }
  return _lerp_tensor(
      start, end, weight, "aten::lerp.Tensor");
}

Tensor& lerp_tensor_(Tensor& self, const Tensor& end, const Tensor& weight) {
  if (weight.sizes().size() == 0) {
    return _lerp_scalar_(
        self, end, weight.item<float>(), "aten::lerp_.Tensor");
  }
  return _lerp_tensor_(
      self, end, weight, "aten::lerp_.Tensor");
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::lerp.Scalar"), TORCH_FN(lerp_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::lerp_.Scalar"), TORCH_FN(lerp_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::lerp.Tensor"), TORCH_FN(lerp_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::lerp_.Tensor"), TORCH_FN(lerp_tensor_));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
