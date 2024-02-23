#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor permute_4d(
    const Tensor& input_arg,
    const uvec4& in_size,
    const uvec4& out_size,
    const uvec4& out_dims,
    vTensor& v_output) {
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_self = convert(input);

  uint32_t out_channels = out_size.data[1u];
  uint32_t in_channels = in_size.data[1u];

  uint32_t out_c_aligned = api::utils::align_up(out_channels, 4u);
  uint32_t in_c_aligned = api::utils::align_up(in_channels, 4u);

  const struct Block final {
    ivec3 out_extents;
    int32_t fill0;
    ivec3 in_extents;
    int32_t fill1;
    uvec4 out_tensor_size;
    uvec4 in_tensor_size;
    uvec4 out_ndims;
    uvec2 ch_info;
  } block{
      api::utils::make_ivec3(v_output.extents()),
      0,
      api::utils::make_ivec3(v_self.extents()),
      0,
      out_size,
      in_size,
      out_dims,
      {out_c_aligned, in_c_aligned},
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(permute_4d),
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
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor permute(const Tensor& self, IntArrayRef dims) {
  auto nDims = safe_downcast<uint32_t>(self.dim());
  TORCH_CHECK(
      dims.size() == (size_t)nDims, "number of dims don't match in permute");

  uvec4 in_size{1u, 1u, 1u, 1u}, out_size{1u, 1u, 1u, 1u};
  uvec4 out_dims{0u, 1u, 2u, 3u};

  auto oldSizes = self.sizes();
  DimVector newSizes(nDims);
  bool sameDims = true;
  std::vector<bool> seen(nDims);
  for (const auto i : c10::irange(nDims)) {
    auto dim = safe_downcast<uint32_t>(maybe_wrap_dim(dims[i], nDims));
    TORCH_CHECK(!seen[dim], "repeated dim in permute");
    seen[dim] = true;
    newSizes[i] = oldSizes[dim];
    if (dim != i) {
      sameDims = false;
    }
    // generalize into 4D tensor
    in_size.data[(4u - nDims) + i] = self.sizes()[i];
    out_size.data[(4u - nDims) + i] = self.sizes()[dim];
    out_dims.data[(4u - nDims) + i] = dim + (4u - nDims);
  }

  if (sameDims) {
    return self;
  }

  IntArrayRef output_sizes(newSizes);
  vTensor v_output{
      api::context(),
      output_sizes.vec(),
      convert_dtype(self.scalar_type()),
  };

  return permute_4d(self, in_size, out_size, out_dims, v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::permute"), TORCH_FN(permute));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
