#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor flip(const at::Tensor& self, const IntArrayRef dim_list) {
  TORCH_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan flip supports up to 4d tensors as input!");

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Cast the input Tensor to a vTensor
  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  const vTensor& v_input = convert(input);

  // Create the output texture
  vTensor v_output{
      context,
      v_input.sizes(),
      convert_dtype(self.scalar_type()),
  };

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  // Create dim args
  std::vector<int32_t> dim_args = {0, 0, 0, 0};
  for (const auto dim : dim_list) {
    TORCH_CHECK(
        dim >= -self.dim() - 1 && dim <= self.dim(),
        "Vulkan flip dimension out of range expected to be in range of [",
        -self.dim() - 1,
        ",",
        self.dim(),
        "], but got ",
        dim);
    // Normalize
    int normalized_dim = utils::normalize(dim, self.dim());

    // Shift into 4d range
    if (self.dim() < 4) {
      normalized_dim += (4 - self.dim());
    }
    dim_args[normalized_dim] = 1;
  }

  // Create the params buffer
  const struct Block final {
    uvec4 extents;
    ivec4 dims;
  } block{
      {get_dim<Dim4D::Width>(v_output),
       get_dim<Dim4D::Height>(v_output),
       get_dim<Dim4D::Channel>(v_output),
       get_dim<Dim4D::Batch>(v_output)},
      {dim_args[3], dim_args[2], dim_args[1], dim_args[0]},
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(flip),
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
};

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::flip"), TORCH_FN(flip));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
