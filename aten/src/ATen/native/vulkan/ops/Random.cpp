#include <ATen/ArrayRef.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <torch/library.h>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

using namespace api::utils;

#ifdef USE_VULKAN_API

static Tensor& uniform_(
    Tensor& self,
    const double from,
    const double to,
    const c10::optional<at::Generator> /* not implemented */) {
  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self);

  const struct Block final {
    uvec3 extents;
    float from;
    float to;
  } block{v_self.extents(), static_cast<float>(from), static_cast<float>(to)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      // shader_descriptor,
      VK_KERNEL(uniform_),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self;
}

static Tensor rand_like(
    const at::Tensor& input_arg,
    const c10::optional<c10::ScalarType> /* not implemented */,
    const c10::optional<c10::Layout> /* not implemented */,
    const c10::optional<c10::Device> /* not implemented */,
    const c10::optional<bool> /* not implemented */,
    const c10::optional<c10::MemoryFormat> /* not implemented */) {
  // Returns a tensor with the same size as input that is filled with random
  // numbers from a uniform distribution on the interval [0,1). To match the CPU
  // implementation, we simplify the range to [0,1] and tolerate the small
  // chance of 1 being sampled.
  return input_arg.clone().detach().uniform_(0.0, 1.0);
}

static Tensor& normal_(
    Tensor& self,
    const double mean,
    const double std,
    const c10::optional<at::Generator> /* not implemented */) {
  TORCH_CHECK(
      self.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  TORCH_CHECK(std >= 0, "Vulkan: Standard deviation (std) can be negative.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self);

  const struct Block final {
    uvec3 extents;
    float mean;
    float std;
  } block{v_self.extents(), static_cast<float>(mean), static_cast<float>(std)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      // shader_descriptor,
      VK_KERNEL(normal_),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self;
}

static Tensor randn_like(
    const at::Tensor& input_arg,
    const c10::optional<c10::ScalarType> /* not implemented */,
    const c10::optional<c10::Layout> /* not implemented */,
    const c10::optional<c10::Device> /* not implemented */,
    const c10::optional<bool> /* not implemented */,
    const c10::optional<c10::MemoryFormat> /* not implemented */) {
  // Returns a tensor with the same size as input that is filled with random
  // numbers from a normal distribution with mean 0 and standard deviation 1.
  return input_arg.clone().detach().normal_(0.0, 1.0);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::uniform_"), TORCH_FN(uniform_));
  m.impl(TORCH_SELECTIVE_NAME("aten::rand_like"), TORCH_FN(rand_like));
  m.impl(TORCH_SELECTIVE_NAME("aten::normal_"), TORCH_FN(normal_));
  m.impl(TORCH_SELECTIVE_NAME("aten::randn_like"), TORCH_FN(randn_like));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
