#include <ATen/native/vulkan/ops/Tensor.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

api::Resource::Image allocate_image(const IntArrayRef sizes) {
  const size_t dim = sizes.size();

  TORCH_INTERNAL_ASSERT(
      dim <= 4u,
      "Only Tensors with dim <= 4 can be represented as a Vulkan Image!");

  int64_t width = 1;
  int64_t height = 1;
  int64_t depth = 1;

  if (dim == 4) {
    width = sizes[3];
    height = sizes[2];
    depth = sizes[0] * sizes[1];
  } else if (dim == 3) {
    width = sizes[2];
    height = sizes[1];
    depth = sizes[0];
  } else if (dim == 2) {
    width = sizes[1];
    height = sizes[0];
  } else if (dim == 1) {
    width = sizes[0];
  }

  return api::context().resource().pool.allocate(
      api::Resource::Image::Descriptor{
        VK_IMAGE_TYPE_3D,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        {
          width,
          height,
          depth,
        },
        // Usage
        {
          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
          VMA_MEMORY_USAGE_GPU_ONLY,
        },
        // View
        {
          VK_IMAGE_VIEW_TYPE_3D,
          VK_FORMAT_R16G16B16A16_SFLOAT,
        },
      });
}

} // namespace

vTensor::vTensor()
  : image_{} {
}

vTensor::vTensor(const IntArrayRef sizes)
  : image_(allocate_image(sizes)) {
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
