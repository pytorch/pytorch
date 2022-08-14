#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

uint32_t batch_size(const IntArrayRef sizes) {
  const uint32_t dims = sizes.size();
  if (dims < 4) {
    return 1;
  }
  return sizes[dims - 4];
}

uint32_t batch_size(const Tensor& tensor) {
  return batch_size(tensor.sizes());
}

uint32_t channels_size(const IntArrayRef sizes) {
  const uint32_t dims = sizes.size();
  if (dims < 3) {
    return 1;
  }
  return sizes[dims - 3];
}

uint32_t channels_size(const Tensor& tensor) {
  return channels_size(tensor.sizes());
}

uint32_t height_size(const IntArrayRef sizes) {
  const uint32_t dims = sizes.size();
  if (dims < 2) {
    return 1;
  }
  return sizes[dims - 2];
}

uint32_t height_size(const Tensor& tensor) {
  return height_size(tensor.sizes());
}

uint32_t width_size(const IntArrayRef sizes) {
  const uint32_t dims = sizes.size();
  if (dims < 1) {
    return 1;
  }
  return sizes[dims - 1];
}

uint32_t width_size(const Tensor& tensor) {
  return width_size(tensor.sizes());
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
