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
  if (context->gpu().adapter->has_unified_memory()) {
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

  if (!context->gpu().adapter->has_unified_memory()) {
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

void copy_staging_to_buffer(
    api::Command::Buffer command_buffer,
    const api::Resource::Buffer staging,
    const api::Resource::Buffer buffer) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      staging,
      "Invalid Vulkan staging buffer!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      buffer,
      "Invalid Vulkan buffer!");

  command_buffer.copy(
      staging.handle,
      buffer.handle,
      std::min(
          staging.memory.allocation_info.size,
          buffer.memory.allocation_info.size));
}

void copy_buffer_to_staging(
    api::Command::Buffer command_buffer,
    const api::Resource::Buffer buffer,
    const api::Resource::Buffer staging) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      buffer,
      "Invalid Vulkan buffer!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      staging,
      "Invalid Vulkan staging buffer!");
}

void copy_buffer_to_image(
    api::Command::Buffer command_buffer,
    const api::Resource::Buffer buffer,
    const api::Resource::Image image) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      buffer,
      "Invalid Vulkan buffer!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      image,
      "Invalid Vulkan image!");
}

void copy_image_to_buffer(
    api::Command::Buffer command_buffer,
    const api::Resource::Image image,
    const api::Resource::Buffer buffer) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      image,
      "Invalid Vulkan image!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      buffer,
      "Invalid Vulkan buffer!");
}

} // namespace

vTensor::vTensor()
  : context_{},
    image_{},
    buffer_{},
    staging_{},
    dirty_{} {
  enforce_invariants();
}

vTensor::vTensor(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options)
  : context_(context),
    image_(maybe_allocate_image(context, sizes, options)),
    buffer_(allocate_buffer(context, sizes, options)),
    staging_(maybe_allocate_staging(context, sizes, options)),
    dirty_{} {
  enforce_invariants();
}

const vTensor* vTensor::host_impl() const {
  enforce_invariants();

  if (dirty_.image || dirty_.buffer) {
    api::Command::Buffer command_buffer =
        context_->command().pool.primary.allocate();

    command_buffer.begin();
    {
      if (dirty_.image) {
        copy_image_to_buffer(command_buffer, image_, buffer_);
        dirty_.image = 0u;

        if (staging_) {
          dirty_.buffer = 1u;
        }
      }

      if (dirty_.buffer && staging_) {
        copy_buffer_to_staging(command_buffer, buffer_, staging_);
        dirty_.buffer = 0u;
      }
    }
    command_buffer.end();
    command_buffer.submit(context_->gpu().queue, VK_NULL_HANDLE);
  }

  return this;
}

vTensor* vTensor::host_impl(const Access::Flags access) {
  vTensor* const tensor = const_cast<vTensor*>(
      const_cast<const vTensor&>(*this).host_impl());

  if (access & Access::Write) {
    if (staging_) {
      dirty_.staging = 1u;
    }
    else {
      dirty_.buffer = 1u;
    }
  }

  return tensor;
}

api::Resource::Memory& vTensor::wait_impl(const Access::Flags access) {
  enforce_invariants();

  api::Resource::Buffer& buffer = staging_ ? staging_ : buffer_;
  TORCH_CHECK(buffer, "Invalid Vulkan buffer!");

  return buffer_.memory;
}

VkBuffer vTensor::buffer() const & {
  enforce_invariants();

  if (dirty_.staging || dirty_.image) {
    api::Command::Buffer command_buffer =
        context_->command().pool.primary.allocate();

    command_buffer.begin();
    {
      if (dirty_.staging) {
        copy_staging_to_buffer(command_buffer, staging_, buffer_);
        dirty_.staging = 0u;
      }
      else if (dirty_.image) {
        copy_image_to_buffer(command_buffer, image_, buffer_);
        dirty_.image = 0u;
      }
    }
    command_buffer.end();
    command_buffer.submit(context_->gpu().queue, VK_NULL_HANDLE);
  }

  return buffer_.handle;
}

VkBuffer vTensor::buffer(const Access::Flags access) & {
  const VkBuffer buffer = const_cast<const vTensor&>(*this).buffer();

  if (buffer && (access & Access::Write)) {
    dirty_.buffer = 1;
  }

  return buffer;
}

VkImage vTensor::image() const & {
  enforce_invariants();

  if (dirty_.staging || dirty_.buffer) {
    api::Command::Buffer command_buffer =
        context_->command().pool.primary.allocate();

    command_buffer.begin();
    {
      if (dirty_.staging) {
        copy_staging_to_buffer(command_buffer, staging_, buffer_);
        dirty_.staging = 0u;
        dirty_.buffer = 1u;
      }

      if (dirty_.buffer) {
        copy_buffer_to_image(command_buffer, buffer_, image_);
        dirty_.buffer = 0u;
      }
    }
    command_buffer.end();
    command_buffer.submit(context_->gpu().queue, VK_NULL_HANDLE);
  }

  return image_.handle;
}

VkImage vTensor::image(const Access::Flags access) & {
  const VkImage image = const_cast<const vTensor&>(*this).image();

  if (image && (access & Access::Write)) {
    dirty_.image = 1;
  }

  return image;
}

void vTensor::enforce_invariants() const {
  TORCH_INTERNAL_ASSERT(!context_ || (context_ && buffer_));
  TORCH_INTERNAL_ASSERT(!dirty_.image || (image_ && dirty_.image));
  TORCH_INTERNAL_ASSERT(!dirty_.buffer || (buffer_ && dirty_.buffer));
  TORCH_INTERNAL_ASSERT(!dirty_.staging || (staging_ && dirty_.staging));
  TORCH_INTERNAL_ASSERT(
      !(dirty_.image || dirty_.buffer || dirty_.staging) ||
      !(dirty_.image ^ dirty_.buffer ^ dirty_.staging));
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
