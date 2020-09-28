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

vTensor::Buffer allocate_buffer(
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
      vTensor::Buffer::Descriptor{
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

bool requires_image(const IntArrayRef sizes) {
  return (1u <= sizes.size()) && (sizes.size() <= 4u);
}

vTensor::Image allocate_image(
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
      vTensor::Image::Descriptor{
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

bool requires_staging(api::Context* const context) {
  return !context->gpu().adapter->has_unified_memory();
}

vTensor::Buffer allocate_staging(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options) {
  TORCH_CHECK(!sizes.empty(), "Invalid Vulkan tensor size!");
  verify(options);

  return context->resource().pool.buffer(
      vTensor::Buffer::Descriptor{
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

vTensor::Fence allocate_fence(
    api::Context* const context) {
  return context->resource().pool.fence();
}

} // namespace

vTensor::vTensor(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options)
  : view_(context, sizes, options) {
}

const vTensor* vTensor::host() const {
  view_.transition({
    View::Component::Staging,
    Memory::Access::Read,
  });

  return this;
}

vTensor* vTensor::host(const Access::Flags access) {
  view_.transition({
    View::Component::Staging,
    access,
  });

  return this;
}

vTensor::Memory& vTensor::wait() {
  //

  return view_.staging().memory;
}

vTensor::Buffer::Object vTensor::buffer() const & {
  view_.transition({
    View::Component::Buffer,
    Memory::Access::Read,
  });

  return view_.buffer().object;
}

vTensor::Buffer::Object vTensor::buffer(const Access::Flags access) & {
  view_.transition({
    View::Component::Buffer,
    access,
  });

  return view_.buffer().object;
}

vTensor::Image::Object vTensor::image() const & {
  view_.transition({
    View::Component::Image,
    Memory::Access::Read,
  });

  return view_.image().object;
}

vTensor::Image::Object vTensor::image(const Access::Flags access) & {
  view_.transition({
    View::Component::Image,
    access,
  });

  return view_.image().object;
}

vTensor::View::View(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options)
  : context_(context),
    image_{},
    buffer_{},
    staging_{},
    fence_{},
    required_{},
    active_{},
    sizes_(sizes),
    options_(options) {
  verify();

  if (requires_image(sizes_)) {
    required_ |= Component::Image;
  }

  if (requires_staging(context_)) {
    required_ |= Component::Staging;
  }
}

vTensor::Buffer& vTensor::View::buffer() const {
  if (!buffer_) {
    buffer_ = allocate_buffer(
        context_,
        sizes_,
        options_);
  }

  return buffer_;
}

vTensor::Image& vTensor::View::image() const {
  if (!image_ && (required_ & Component::Image)) {
    image_ = allocate_image(
        context_,
        sizes_,
        options_);
  }

  return image_;
}

vTensor::Buffer& vTensor::View::staging() const {
  if (!(required_ & Component::Staging)) {
    return buffer();
  }

  if (!staging_) {
    staging_ = allocate_staging(
        context_,
        sizes_,
        options_);
  }

  return staging_;
}

vTensor::Fence& vTensor::View::fence() const {
  if (!fence_) {
    fence_ = allocate_fence(context_);
  }

  return fence_;
}

// Any state in column T can transition to any state in column T + 1.  That
// leaves us with 6 x 6 = 36 possible transitions.  In each scenario,
// synchronization must be handled appropriately.
//
//      T              T + 1
//
// Read  Staging |  Read  Staging
//
// Write Staging |  Write Staging
//
// Read  Buffer  |  Read  Buffer
//
// Write Buffer  |  Write Buffer
//
// Read  Image   |  Read  Image
//
// Write Image   |  Write Image
//

void vTensor::View::transition(const Active view) const {
  verify();

  // Always make sure to update the active view regardless of codepath taken.

  struct Update final {
    const Active& src;
    Active& dst;

    inline ~Update() {
      dst = src;
    }
  } update {
      view,
      active_,
  };

  // Memory availability and visibility operations on host is handled through
  // map() and unmap() if not dealing with coherent host memory.  Other than
  // that, host to host dependencies require no device-side synchronization.
  // This is regardless of whether we are dealing with UMA or discrete systems.
  // That leaves us with 36 - 4 = 32 possible transitions.

  if ((active_.component == Component::Staging) &&
      (view.component == Component::Staging)) {
    return;
  }

  // RAR (Read after Read) is not a hazard.  That leaves us with 32 - 8 = 24
  // possible transitions.

  if ((0u == (active_.access & Memory::Access::Read)) &&
      (0u == (view.access & Memory::Access::Read))) {
    return;
  }

  // All transitions after this point require an explicit synchronization.

  api::Command::Buffer command_buffer = context_->command().pool.allocate();
  command_buffer.begin();

  // WAR (Write after Read) hazards do not need a memory barrier. Execution
  // barriers are sufficient.  This section handles the following 8 WAR
  // transitions, leaving us with 24 - 8 = 16 transitions remaining.

  // Read Staging -> Write Buffer
  // Read Staging -> Write Image
  // Read Buffer  -> Write Staging
  // Read Buffer  -> Write Buffer
  // Read Buffer  -> Write Image
  // Read Image   -> Write Staging
  // Read Image   -> Write Buffer
  // Read Image   -> Write Image

  const auto stage = [](const Active view) -> VkPipelineStageFlags {
    switch(view.component) {
      case Component::Buffer:
      case Component::Image:
        return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

      case Component::Staging:
        return VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
  };

  if ((0u == (active_.access & Memory::Access::Read)) &&
      // Notice how we include Read-Writes, in addition to Writes, as a
      // Write operation in the condition below as well, as we should.
      (active_.access & Memory::Access::Write)) {
  }

  // Handle the remaining 16 RAW or WAW hazards.  Additionally, if transitioning
  // to an Image, handle the layout transition accordingly as well.

  // Write Staging -> Read  Buffer
  // Write Staging -> Read  Image
  // Write Staging -> Write Buffer
  // Write Staging -> Write Image
  // ---
  // Write Buffer  -> Read  Staging
  // Write Buffer  -> Write Staging
  // Write Buffer  -> Read  Buffer
  // Write Buffer  -> Read  Image
  // Write Buffer  -> Write Buffer
  // Write Buffer  -> Write Image
  // ---
  // Write Image   -> Read  Staging
  // Write Image   -> Write Staging
  // Write Image   -> Read  Buffer
  // Write Image   -> Read  Image
  // Write Image   -> Write Buffer
  // Write Image   -> Write Image

  else {

  }

  command_buffer.end();
  command_buffer.submit(context_->gpu().queue);
}

void vTensor::View::verify() const {
  TORCH_INTERNAL_ASSERT(!image_ || (required_ & Component::Image));
  TORCH_INTERNAL_ASSERT(!staging_ || (required_ & Component::Staging));

  // TORCH_INTERNAL_ASSERT(
  //     buffer ||
  //     !dirty(View::Resource::Buffer));

  // TORCH_INTERNAL_ASSERT(
  //     data_.image ||
  //     !data_.dirty(View::Resource::Image));

  // TORCH_INTERNAL_ASSERT(
  //     data_.staging ||
  //     !data_.dirty(View::Resource::Staging));

  // TORCH_INTERNAL_ASSERT(
  //     !data_.dirty(
  //         View::Resource::Buffer |
  //         View::Resource::Image |
  //         View::Resource::Staging) ||
  //     ((data_.dirty(View::Resource::Buffer) !=
  //       data_.dirty(View::Resource::Image)) !=
  //       data_.dirty(View::Resource::Staging)));
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
