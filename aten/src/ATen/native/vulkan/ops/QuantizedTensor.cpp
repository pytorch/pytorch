#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <sys/_types/_int64_t.h>
#include <sys/_types/_u_int32_t.h>
#include <torch/library.h>
#include "c10/core/ScalarType.h"

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor quantize_per_tensor(
    const at::Tensor& input_arg,
    const double scale,
    const int64_t zero_point,
    const c10::ScalarType dtype) {

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
    api::OpProfiler profiler(command_buffer, context->querypool(), "aten::quantize_per_tensor");

    if C10_LIKELY(v_input.has_image()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scale;
        float _1;
        int32_t zero_point;
        int32_t _2;
      } block {
        v_output.extents(),
        0u,
        safe_downcast<float>(scale),
        0.0f,
        safe_downcast<int32_t>(zero_point),
        0u,
      };


      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(quantize_per_tensor),
          v_input.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_input.image(
              command_buffer,
              vTensor::Stage::Compute),
          // Object lifetime is managed by the resource pool.
          // It is OK not to keep track of the handle.
          context->resource().pool.uniform(block).object);
    } else {
      TORCH_CHECK(false, "Not implemented!");
    }
  }
  command_pool.submit(context->gpu().queue, command_buffer);
  v_output.q_scale = scale;
  v_output.q_zero_point = zero_point;
  return convert(v_output);
}

//helper for dequantize function to use scale and zero_point
Tensor dequantize_helper(
    const at::Tensor& input_arg,
    const double scale,
    const int64_t zero_point,
    const c10::ScalarType dtype) {

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
    api::OpProfiler profiler(command_buffer, context->querypool(), "aten::dequantize.self");

    if C10_LIKELY(v_input.has_image()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scale;
        float _1;
        int32_t zero_point;
        int32_t _2;
      } block {
        v_output.extents(),
        0u,
        safe_downcast<float>(scale),
        0.0f,
        safe_downcast<int32_t>(zero_point),
        0u,
      };


      context->dispatch(
          command_buffer,
          {
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          VK_KERNEL(dequantize),
          v_input.extents(),
          context->gpu().adapter->local_work_group_size(),
          // Write-only access bypasses synchronization but inserts appropriate
          // barriers if necessary.
          v_output.image(
              command_buffer,
              vTensor::Stage::Compute,
              vTensor::Access::Write),
          // Read-only access is implied on const tensors and triggers an async
          // synchronization if necessary.
          v_input.image(
              command_buffer,
              vTensor::Stage::Compute),
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

Tensor dequantize(
    const Tensor& self) {
        //get params needed to dequantize back as they are not inputted
        double q_scale = convert(self).q_scale;
        int zero_point = convert(self).q_zero_point;
  return dequantize_helper(self, q_scale, zero_point, kFloat);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::quantize_per_tensor"), quantize_per_tensor);
  m.impl(TORCH_SELECTIVE_NAME("aten::dequantize.self"), dequantize);
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
