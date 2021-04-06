#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

uint32_t batch_size(const Tensor& tensor) {
  const IntArrayRef sizes = tensor.sizes();
  const uint32_t dims = sizes.size();
  if (dims < 4) {
    return 1;
  }
  return sizes[dims - 4];
}

uint32_t channels_size(const Tensor& tensor) {
  const IntArrayRef sizes = tensor.sizes();
  const uint32_t dims = sizes.size();
  if (dims < 3) {
    return 1;
  }
  return sizes[dims - 3];
}

uint32_t height_size(const Tensor& tensor) {
  const IntArrayRef sizes = tensor.sizes();
  const uint32_t dims = sizes.size();
  if (dims < 2) {
    return 1;
  }
  return sizes[dims - 2];
}

uint32_t width_size(const Tensor& tensor) {
  const IntArrayRef sizes = tensor.sizes();
  const uint32_t dims = sizes.size();
  if (dims < 1) {
    return 1;
  }
  return sizes[dims - 1];
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
