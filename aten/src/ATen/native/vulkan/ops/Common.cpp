#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

uint32_t batch_size(const Tensor& tensor) {
  uint32_t dims = tensor.sizes().size();
  if (dims < 4) {
    return 1;
  }
  return tensor.sizes()[dims - 4];
}

uint32_t channels_size(const Tensor& tensor) {
  uint32_t dims = tensor.sizes().size();
  if (dims < 3) {
    return 1;
  }
  return tensor.sizes()[dims - 3];
}

uint32_t height_size(const Tensor& tensor) {
  uint32_t dims = tensor.sizes().size();
  if (dims < 2) {
    return 1;
  }
  return tensor.sizes()[dims - 2];
}

uint32_t width_size(const Tensor& tensor) {
  uint32_t dims = tensor.sizes().size();
  if (dims < 1) {
    return 1;
  }
  return tensor.sizes()[dims - 1];
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
