#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor sum_dim(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    const optional<ScalarType> dtype) {
  TORCH_CHECK(
      self.dim() >= 2 || self.dim() <= 4,
      "Vulkan sum.dim_IntList supports 2d, 3d, 4d tensors as input!");
  TORCH_CHECK(
      dim >= -self.dim() - 1 && dim <= self.dim(),
      "Vulkan sum.dim_IntList dimension out of range expected to be in range of [",
      -self.dim() - 1,
      ",",
      self.dim(),
      "], but got ",
      dim);

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Cast the input Tensor to a vTensor
  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  const vTensor& v_input = convert(input);

  // Normalize dim into range [0, self.dim()]
  dim = utils::normalize(dim, self.dim());

  // Create the output texture
  std::vector<int64_t> output_size = self.sizes().vec();
  uint32_t dim_size = output_size[dim];
  output_size.erase(output_size.begin() + dim);

  ScalarType type = self.scalar_type();
  if (dtype.has_value()) {
    type = dtype.value();
  }

  vTensor v_output{
      context,
      output_size,
      type,
  };

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  // Shift dim into 4d range
  if (self.dim() < 4) {
    dim += (4 - self.dim());
  }

  // Create the params buffer
  const struct Block final {
    uvec2 dim_info;
    int32_t channel;
  } block{
      {static_cast<uint32_t>(dim), dim_size},
      static_cast<int32_t>(get_dim<Dim4D::Channel>(v_input)),
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(sum_dim),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());
  return convert(v_output);
}

Tensor sum_dim_IntList(
    const at::Tensor& self,
    const OptionalIntArrayRef opt_dim,
    bool keepdim,
    const optional<ScalarType> dtype) {
  TORCH_CHECK(
      opt_dim.has_value(),
      "Vulkan sum.dim_IntList without a dim arg is not implemented");
  TORCH_CHECK(
      keepdim == false,
      "Vulkan sum.dim_IntList with keepdim=true is not implemented");

  std::set<int64_t> dims_set;
  if (opt_dim.has_value()) {
    auto dims = opt_dim.value();
    for (const auto& d : dims) {
      dims_set.insert(d);
    }
    Tensor result = self;
    for (auto it = dims_set.rbegin(); it != dims_set.rend(); ++it) {
      result = sum_dim(result, *it, keepdim, dtype);
    }
    return result;
  }
  return self;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::sum.dim_IntList"), TORCH_FN(sum_dim_IntList));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
