#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

api::utils::uvec4 make_sizes_uvec4(const IntArrayRef sizes) {
  uint32_t width = get_dim<Dim4D::Width>(sizes);
  uint32_t height = get_dim<Dim4D::Height>(sizes);
  uint32_t channels = get_dim<Dim4D::Channel>(sizes);
  uint32_t batches = get_dim<Dim4D::Batch>(sizes);

  return {width, height, channels, batches};
}

api::utils::uvec4 make_strides_uvec4(const IntArrayRef strides) {
  uint32_t w_stride = get_dim<Dim4D::Width>(strides);
  uint32_t h_stride = get_dim<Dim4D::Height>(strides);
  uint32_t c_stride = get_dim<Dim4D::Channel>(strides);
  uint32_t n_stride = get_dim<Dim4D::Batch>(strides);

  return {w_stride, h_stride, c_stride, n_stride};
}

api::utils::uvec3 adaptive_work_group_size(
    const api::utils::uvec3& global_work_group) {
  api::utils::uvec3 local_group_size = {4, 4, 4};
  if (global_work_group.data[2u] == 1) {
    if (global_work_group.data[1u] < 8) {
      local_group_size.data[0u] = 16;
      local_group_size.data[1u] = 4;
      local_group_size.data[2u] = 1;
    } else {
      local_group_size.data[0u] = 8;
      local_group_size.data[1u] = 8;
      local_group_size.data[2u] = 1;
    }
  }
  return local_group_size;
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
