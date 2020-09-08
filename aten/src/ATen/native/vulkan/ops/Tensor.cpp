#include <ATen/native/vulkan/ops/Tensor.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

bool can_be_image(const IntArrayRef sizes) {
  return (1u <= sizes.size()) && (sizes.size() <= 4u);
}

VkFormat convert(const caffe2::TypeMeta dtype) {
  switch (c10::typeMetaToScalarType(dtype)) {
    case kFloat:
      // VK_FORMAT_R32G32B32A32_SFLOAT?
      return VK_FORMAT_R16G16B16A16_SFLOAT;

    default:
      TORCH_CHECK(
        false,
        "Vulkan tensor format not supported!");
  }

  return VK_FORMAT_UNDEFINED;
}

api::Resource::Buffer allocate_staging(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options) {
  TORCH_CHECK(!sizes.empty(), "Invalid Vulkan tensor size!");
  verify(options);

  return context->resource().pool.allocate(
      api::Resource::Buffer::Descriptor{
        std::accumulate(
            sizes.cbegin(),
            sizes.cend(),
            1,
            std::multiplies<int64_t>()),
        // Usage
        {
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          VMA_MEMORY_USAGE_CPU_ONLY,
        },
      });
}

api::Resource::Buffer maybe_allocate_staging(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options) {
  if (context->adapter().has_unified_memory()) {
    return api::Resource::Buffer{};
  }

  return allocate_staging(context, sizes, options);
}

api::Resource::Buffer allocate_buffer(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options) {
  TORCH_CHECK(!sizes.empty(), "Invalid Vulkan tensor size!");
  verify(options);

  VkFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  if (can_be_image(sizes)) {
    usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }

  if (!context->adapter().has_unified_memory()) {
    usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }

  return context->resource().pool.allocate(
      api::Resource::Buffer::Descriptor{
        std::accumulate(
            sizes.cbegin(),
            sizes.cend(),
            1,
            std::multiplies<int64_t>()),
        // Usage
        {
          usage,
          VMA_MEMORY_USAGE_GPU_ONLY,
        },
      });
}

api::Resource::Buffer maybe_allocate_buffer(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options) {
  // Always need a buffer.
  return allocate_buffer(context, sizes, options);
}

api::Resource::Image allocate_image(
    api::Context* const context,
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

  return context->resource().pool.allocate(
      api::Resource::Image::Descriptor{
        VK_IMAGE_TYPE_3D,
        convert(options.dtype()),
        {
          width,
          height,
          depth,
        },
        // Usage
        {
          VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_STORAGE_BIT,
          VMA_MEMORY_USAGE_GPU_ONLY,
        },
        // View
        {
          VK_IMAGE_VIEW_TYPE_3D,
          convert(options.dtype()),
        },
      });
}

api::Resource::Image maybe_allocate_image(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options) {
  if (!can_be_image(sizes)) {
    return api::Resource::Image{};
  }

  return allocate_image(context, sizes, options);
}

} // namespace

vTensor::vTensor()
  : image_{},
    buffer_{},
    staging_{},
    context_{},
    dirty_{} {
}

vTensor::vTensor(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options)
  : image_(maybe_allocate_image(context, sizes, options)),
    buffer_(maybe_allocate_buffer(context, sizes, options)),
    staging_(maybe_allocate_staging(context, sizes, options)),
    context_(context),
    sizes_(sizes.cbegin(), sizes.cend()),
    options_(options),
    dirty_{} {
}

VkBuffer vTensor::buffer() const {
  if (dirty_.image) {
    //

    dirty_.image = 0u;
  }

  if (dirty_.staging) {
    //

    dirty_.staging = 0u;
  }

  return buffer_.handle;
}

VkBuffer vTensor::buffer(const Access::Flags access) {
  const VkBuffer buffer = const_cast<const vTensor&>(*this).buffer();

  if (access & Access::Write) {
    dirty_.buffer = 1;
  }

  return buffer;
}

VkImage vTensor::image() const {
  if (dirty_.buffer) {
    //

    dirty_.buffer = 0u;
  }

  return image_.handle;
}

VkImage vTensor::image(const Access::Flags access) {
  const VkImage image = const_cast<const vTensor&>(*this).image();

  if (access & Access::Write) {
    dirty_.image = 1;
  }

  return image;
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
