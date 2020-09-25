#include <ATen/native/vulkan/ops/Tensor.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

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

bool should_have_image(const IntArrayRef sizes) {
  return (1u <= sizes.size()) && (sizes.size() <= 4u);
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

  return context->resource().pool.image(
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

api::Resource::Buffer allocate_buffer(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options) {
  TORCH_CHECK(!sizes.empty(), "Invalid Vulkan tensor size!");
  verify(options);

  VkFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  if (!context->gpu().adapter->has_unified_memory()) {
    usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }

  return context->resource().pool.buffer(
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

bool should_have_staging(api::Context* const context) {
  return !context->gpu().adapter->has_unified_memory();
}

api::Resource::Buffer allocate_staging(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options) {
  TORCH_CHECK(!sizes.empty(), "Invalid Vulkan tensor size!");
  verify(options);

  return context->resource().pool.buffer(
      api::Resource::Buffer::Descriptor{
        std::accumulate(
            sizes.cbegin(),
            sizes.cend(),
            1,
            std::multiplies<int64_t>()),
        // Usage
        {
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VMA_MEMORY_USAGE_CPU_ONLY,
        },
      });
}

api::Resource::Fence allocate_fence(
    api::Context* const context) {
  return context->resource().pool.fence();
}

void copy_staging_to_buffer(
    api::Command::Buffer& command_buffer,
    const vTensor::Buffer& staging,
    const vTensor::Buffer& buffer) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      staging,
      "Invalid Vulkan staging buffer!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      buffer,
      "Invalid Vulkan buffer!");

  command_buffer.copy(
      staging.handle,
      buffer.handle,
      std::min(staging.range, buffer.range));
}

void copy_buffer_to_staging(
    api::Command::Buffer& command_buffer,
    const vTensor::Buffer& buffer,
    const vTensor::Buffer& staging) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      buffer,
      "Invalid Vulkan buffer!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      staging,
      "Invalid Vulkan staging buffer!");

  command_buffer.copy(
      buffer.handle,
      staging.handle,
      std::min(staging.range, buffer.range));
}

void copy_buffer_to_image(
    api::Command::Buffer& command_buffer,
    const vTensor::Buffer& buffer,
    const vTensor::Image& image) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      buffer,
      "Invalid Vulkan buffer!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      image,
      "Invalid Vulkan image!");
}

void copy_image_to_buffer(
    api::Command::Buffer& command_buffer,
    const vTensor::Image& image,
    const vTensor::Buffer& buffer) {
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
    view_{} {
  enforce_invariants();
}

vTensor::vTensor(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options)
  : context_(context),
    view_{},
    sizes_(sizes),
    options_(options) {
  enforce_invariants();

  view_.should_have.image = should_have_image(sizes_);
  view_.should_have.staging = should_have_staging(context_);
}

const vTensor* vTensor::host_impl() const {
  enforce_invariants();

  if (view_.dirty.image || view_.dirty.buffer) {
    api::Command::Buffer command_buffer =
        context_->command().pool.buffer();

    command_buffer.begin();
    {
      if (view_.dirty.image) {
        copy_image_to_buffer(
            command_buffer,
            image_().object,
            buffer_().object);

        view_.dirty.image = 0u;

        if (staging_()) {
          view_.dirty.buffer = 1u;
        }
      }

      if (view_.dirty.buffer && staging_()) {
        copy_buffer_to_staging(
            command_buffer,
            buffer_().object,
            staging_().object);

        view_.dirty.buffer = 0u;
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
    if (staging_()) {
      view_.dirty.staging = 1u;
    }
    else {
      view_.dirty.buffer = 1u;
    }
  }

  return tensor;
}

api::Resource::Memory& vTensor::wait_impl() {
  enforce_invariants();

  api::Resource::Buffer& buffer = staging_() ? staging_() : buffer_();
  TORCH_CHECK(buffer, "Invalid Vulkan buffer!");

  return buffer.memory;
}

vTensor::Buffer vTensor::buffer() const & {
  enforce_invariants();

  if (view_.dirty.staging || view_.dirty.image) {
    api::Command::Buffer command_buffer =
        context_->command().pool.buffer();

    command_buffer.begin();
    {
      if (view_.dirty.staging) {
        copy_staging_to_buffer(
            command_buffer,
            staging_().object,
            buffer_().object);

        view_.dirty.staging = 0u;
      }
      else if (view_.dirty.image) {
        copy_image_to_buffer(
            command_buffer,
            image_().object,
            buffer_().object);

        view_.dirty.image = 0u;
      }
    }
    command_buffer.end();
    command_buffer.submit(context_->gpu().queue, VK_NULL_HANDLE);
  }

  return buffer_().object;
}

vTensor::Buffer vTensor::buffer(const Access::Flags access) & {
  const vTensor::Buffer buffer = const_cast<const vTensor&>(*this).buffer();

  if (buffer && (access & Access::Write)) {
    view_.dirty.buffer = 1;
  }

  return buffer;
}

vTensor::Image vTensor::image() const & {
  enforce_invariants();

  if (view_.dirty.staging || view_.dirty.buffer) {
    api::Command::Buffer command_buffer =
        context_->command().pool.buffer();

    command_buffer.begin();
    {
      if (view_.dirty.staging) {
        copy_staging_to_buffer(
            command_buffer,
            staging_().object,
            buffer_().object);

        view_.dirty.staging = 0u;
        view_.dirty.buffer = 1u;
      }

      if (view_.dirty.buffer) {
        copy_buffer_to_image(
            command_buffer,
            buffer_().object,
            image_().object);

        view_.dirty.buffer = 0u;
      }
    }
    command_buffer.end();
    command_buffer.submit(context_->gpu().queue, VK_NULL_HANDLE);
  }

  return image_().object;
}

vTensor::Image vTensor::image(const Access::Flags access) & {
  const vTensor::Image image = const_cast<const vTensor&>(*this).image();

  if (image && (access & Access::Write)) {
    view_.dirty.image = 1;
  }

  return image;
}

void vTensor::enforce_invariants() const {
  TORCH_INTERNAL_ASSERT(view_.image || !view_.dirty.image);
  TORCH_INTERNAL_ASSERT(!view_.image || view_.should_have.image);
  TORCH_INTERNAL_ASSERT(!(view_.buffer && view_.dirty.buffer));
  TORCH_INTERNAL_ASSERT(view_.staging || !view_.dirty.staging);
  TORCH_INTERNAL_ASSERT(!view_.staging || view_.should_have.staging);
  TORCH_INTERNAL_ASSERT(
      !(view_.dirty.image || view_.dirty.buffer || view_.dirty.staging) ||
      (view_.dirty.image ^ view_.dirty.buffer ^ view_.dirty.staging));
}

api::Resource::Image& vTensor::image_() const {
  if (!view_.image && view_.should_have.image) {
    view_.image = allocate_image(
        context_,
        sizes_,
        options_);
  }

  return view_.image;
}

api::Resource::Buffer& vTensor::buffer_() const {
  if (!view_.buffer) {
    view_.buffer = allocate_buffer(
        context_,
        sizes_,
        options_);
  }

  return view_.buffer;
}

api::Resource::Buffer& vTensor::staging_() const {
  if (!view_.staging && view_.should_have.staging) {
    view_.staging = allocate_staging(
        context_,
        sizes_,
        options_);
  }

  return view_.staging;
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
