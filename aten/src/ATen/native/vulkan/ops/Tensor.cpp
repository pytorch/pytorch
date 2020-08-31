#include <ATen/native/vulkan/ops/Tensor.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

VkFormat convert(const caffe2::TypeMeta dtype) {
  switch (c10::typeMetaToScalarType(dtype)) {
    case kFloat:
      return VK_FORMAT_R16G16B16A16_SFLOAT;

    default:
      TORCH_CHECK(
        false,
        "Vulkan tensor format not supported!");
  }

  return VK_FORMAT_UNDEFINED;
}

api::Resource::Buffer allocate_buffer(
    const IntArrayRef sizes,
    const TensorOptions& options) {
  TORCH_INTERNAL_ASSERT(!sizes.empty(), "Invalid Vulkan tensor size!");
  verify(options);

  return api::context().resource().pool.allocate(
      api::Resource::Buffer::Descriptor{
        std::accumulate(
            sizes.cbegin(),
            sizes.cend(),
            1,
            std::multiplies<int64_t>()),
        // Usage
        {
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          VMA_MEMORY_USAGE_GPU_ONLY,
        },
      });
}

api::Resource::Buffer maybe_allocate_buffer(
    const IntArrayRef sizes,
    const TensorOptions& options) {
  if (sizes.size() <= 4u) {
    return api::Resource::Buffer{};
  }

  return allocate_buffer(size, options);
}

api::Resource::Image allocate_image(
    const IntArrayRef sizes,
    const TensorOptions& options) {
  verify(options);

  int64_t width = 1;
  int64_t height = 1;
  int64_t depth = 1;

  switch (sizes.size()) {
    case 1:
      width = sizes[0];
      break;

    case 2:
      width = sizes[1];
      height = sizes[0];
      break;

    case 3:
      width = sizes[2];
      height = sizes[1];
      depth = sizes[0];
      break;

    case 4:
      width = sizes[3];
      height = sizes[2];
      depth = sizes[0] * sizes[1];
      break;

    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Only Tensors with 1 <= dim <= 4 can be represented as a Vulkan Image!");
  }

  const VkFormat format = convert(options.dtype());

  return api::context().resource().pool.allocate(
      api::Resource::Image::Descriptor{
        VK_IMAGE_TYPE_3D,
        format,
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
          format,
        },
      });
}

api::Resource::Image maybe_allocate_image(
    const IntArrayRef sizes,
    const TensorOptions& options) {
  if (sizes.size() > 4u) {
    return api::Resource::Image{};
  }

  return allocate_image(sizes, options);
}

} // namespace

vTensor::vTensor()
  : buffer_{},
    image_{} {
}

vTensor::vTensor(const IntArrayRef sizes, const TensorOptions& options)
  : sizes_(sizes.cbegin(), sizes.cend()),
    options_(options),
    buffer_(maybe_allocate_buffer(sizes, options)),
    image_(maybe_allocate_image(sizes, options)) {
}

void verify(const TensorOptions& options) {
  TORCH_CHECK(
      !options.has_requires_grad() || !options.requires_grad(),
      "'requires_grad' tensor option is not yet supported under Vulkan!");

  TORCH_CHECK(
      !options.has_pinned_memory() || !options.pinned_memory(),
      "'pinned_memory' tensor option is not yet supported under Vulkan!");

  TORCH_CHECK(
      !options.has_layout() || (c10::kStrided == options.layout()),
      "'layout' tensor option is not yet supported under Vulkan!");

  TORCH_CHECK(
      !options.has_memory_format(),
      "'memory_format' tensor option is not yet supported under Vulkan!");
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
