#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor transpose_4d(
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

Tensor transpose(const Tensor& self, int64_t index0, int64_t index1) {
  TORCH_CHECK(
      self.dim() <= 4,
      "Vulkan transpose only supports tensors <= 4 dimensions");

  auto nDims = safe_downcast<uint32_t>(self.dim());
  uvec4 in_size{1u, 1u, 1u, 1u}, out_size{1u, 1u, 1u, 1u};
  uvec4 out_dims{0u, 1u, 2u, 3u};

  auto oldSizes = self.sizes();
  DimVector newSizes(nDims);
  auto new_index0 = safe_downcast<uint32_t>(maybe_wrap_dim(index0, nDims));
  auto new_index1 = safe_downcast<uint32_t>(maybe_wrap_dim(index1, nDims));
  if (new_index0 == new_index1) {
    return self.detach();
  }

  // generalize input and output into 4D tensor, e.g. input is 3d of shape [2,
  // 3, 4] by padding at the batch dim, input becomes 4d with in_size = [1, 2,
  // 3, 4]
  for (const auto i : c10::irange(nDims)) {
    in_size.data[(4u - nDims) + i] = self.sizes()[i];
    out_size.data[(4u - nDims) + i] = self.sizes()[i];
    newSizes[i] = oldSizes[i];
  }

  // get the size of the output by swapping the size of input at index0 and
  // index1 continue with the example above, if index0 = 0, index1 = 2, then
  // output is of size out_size = [1, 4, 3, 2].
  // Note: indices are shifted by (4u - nDims) since input is generalized into
  // 4d.
  out_size.data[(4u - nDims) + new_index0] =
      in_size.data[(4u - nDims) + new_index1];
  out_size.data[(4u - nDims) + new_index1] =
      in_size.data[(4u - nDims) + new_index0];

  // get the desired ordering of dimensions, again we shift by (4u - nDims).
  // Using the example above, out_dims = [0, 3, 2, 1]
  auto temp_dim = out_dims.data[(4u - nDims) + new_index0];
  out_dims.data[(4u - nDims) + new_index0] =
      out_dims.data[(4u - nDims) + new_index1];
  out_dims.data[(4u - nDims) + new_index1] = temp_dim;

  // get the size of the output by swapping sizes of the input. Continue with
  // the example, newSizes = [1, 4, 3, 2]
  newSizes[new_index0] = oldSizes[new_index1];
  newSizes[new_index1] = oldSizes[new_index0];

  IntArrayRef output_size(newSizes);
  vTensor v_output{
      api::context(),
      output_size.vec(),
      convert_dtype(self.scalar_type()),
  };

  return transpose_4d(self, in_size, out_size, out_dims, v_output);
}

Tensor t(const Tensor& self) {
  TORCH_CHECK(self.dim() <= 2, "t() only supports tensors <= 2 dimensions");
  return transpose(self.detach(), 0, self.dim() < 2 ? 0 : 1);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::t"), TORCH_FN(t));
  m.impl(TORCH_SELECTIVE_NAME("aten::transpose.int"), TORCH_FN(transpose));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
