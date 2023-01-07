#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

api::utils::uvec4 make_nchw_uvec4(const IntArrayRef arr) {
  uint32_t w = get_dim<Dim4D::Width>(arr);
  uint32_t h = get_dim<Dim4D::Height>(arr);
  uint32_t c = get_dim<Dim4D::Channel>(arr);
  uint32_t n = get_dim<Dim4D::Batch>(arr);

  return {w, h, c, n};
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
