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
    Access::Read,
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
    Access::Read,
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
    Access::Read,
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

vTensor::View::View()
  : context_(nullptr),
    image_{},
    buffer_{},
    staging_{},
    fence_{},
    required_{},
    active_{} {
  verify();
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
  ops::verify(options);

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
// leaves us with 7 x 6 = 42 possible transitions.  In each scenario,
// synchronization must be handled appropriately.
//
//      T              T + 1
//       Unknown |
// Read  Staging | Read  Staging
// Write Staging | Write Staging
// Read  Buffer  | Read  Buffer
// Write Buffer  | Write Buffer
// Read  Image   | Read  Image
// Write Image   | Write Image
//

void vTensor::View::transition(Active view) const {
  verify();

  // If the target view is an image, either:
  //   1) Make sure image memory is allocated if the tensor can be represented
  //      as an image, or
  //   2) Adjust the target view to a buffer if the tensor cannot be represented
  //      as an image.

  if (Component::Image == view.component) {
    if (required_ & Component::Image) {
      // Force a laze allocation.
      image();
    }
    else {
      // Adjust target view to buffer.
      view.component = Component::Buffer;
    }
  }

  // Always make sure to update the active view and image layout, if necessary,
  // regardless of the codepath taken.  Keep in mind that we are simply updating
  // the state machine here and not issuing any synchronization commands which
  // is what the rest of the logic in this function takes care of if need be.

  struct State final {
    struct {
      Active& view;
      Image::Object& image;

      VkAccessFlags access() const {
        return State::access(view);
      }

      inline VkImageLayout layout() const {
        return image.layout;
      }

      inline void layout(const VkImageLayout layout) {
        image.layout = layout;
      }

      inline VkPipelineStageFlags stage() const {
        return State::stage(view);
      }
    } current;

    const struct {
      Active view;

      VkAccessFlags access() const {
        return State::access(view);
      }

      inline VkImageLayout layout() const {
        return State::layout(view);
      }

      inline VkPipelineStageFlags stage() const {
        return State::stage(view);
      }
    } next;

    inline ~State() {
      current.view = next.view;

      if (Component::Image == next.view.component) {
        current.layout(next.layout());
      }
    }

    bool requires_image_layout_transition() const {
      return (Component::Image == next.view.component) &&
             (next.layout() != current.layout());
    }

    static VkAccessFlags access(const Active view) {
      VkAccessFlags access = 0u;

      switch (stage(view)) {
        case VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT:
          if (view.access & Access::Read) {
            access |= VK_ACCESS_SHADER_READ_BIT;
          }

          if (view.access & Access::Write) {
            access |= VK_ACCESS_SHADER_WRITE_BIT;
          }

          break;

        case VK_PIPELINE_STAGE_TRANSFER_BIT:
          if (view.access & Access::Read) {
            access |= VK_ACCESS_HOST_READ_BIT;
          }

          if (view.access & Access::Write) {
            access |= VK_ACCESS_HOST_WRITE_BIT;
          }

          break;

        default:
          TORCH_INTERNAL_ASSERT(
              false,
              "Invalid Vulkan tensor view state!");
      }

      return access;
    }

    static VkImageLayout layout(const Active view) {
      TORCH_INTERNAL_ASSERT(
          Component::Image == view.component,
          "The active view on the requested Vulkan tensor is not an image!");

      if (view.access & Access::Write) {
        return VK_IMAGE_LAYOUT_GENERAL;
      }

      if (Access::Read == (view.access & Access::Read)) {
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }

      return VK_IMAGE_LAYOUT_UNDEFINED;
    }

    static VkPipelineStageFlags stage(const Active view) {
      switch (view.component) {
        case Component::Buffer:
        case Component::Image:
          return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        case Component::Staging:
          return VK_PIPELINE_STAGE_TRANSFER_BIT;

        default:
          TORCH_INTERNAL_ASSERT(
              false,
              "Invalid Vulkan tensor view state!");
      }
    }
  } state {
      // Current
      {
        active_,
        // Not using the accessor to prevent an unnecessary lazy memory
        // allocation in cases where either
        //   1) The tensor cannot be represented as an image, or
        //   2) The requested target view is not an image.
        image_.object,
      },
      // Next
      {
        view,
      },
  };

  std::cout << "--------------------\n";
  std::cout << "Current - " << state.current.view << std::endl;
  std::cout << "Next - " << state.next.view << std::endl;
  std::cout << "--------------------\n\n";

  // If dealing with a transition from an initial Unknown state, update the
  // active view and return.  No synchronization is reuiqred in such scenarios.
  // This takes care of 6 states originating from an Unknown state leaving us
  // with 42 - 6 = 36 remaining transitions.

  if (Component::Unknown == state.current.view.component) {
    return;
  }

  // Memory availability and visibility operations on host is handled through
  // map() and unmap() if not dealing with coherent host memory.  Other than
  // that, host to host dependencies require no device-side synchronization.
  // This is regardless of whether we are dealing with UMA or discrete systems.
  // This section handles the following 4 transitions which leaves us with
  // 36 - 4 = 32 possible transitions remaining.

  // Read  Staging -> Read  Staging
  // Read  Staging -> Write Staging
  // Write Staging -> Read  Staging
  // Write Staging -> Write Staging

  if ((Component::Staging == state.current.view.component) &&
      (Component::Staging == state.next.view.component)) {
    return;
  }

  // RAR (Read after Read) is not a hazard so no synchronization is required
  // unless we are dealing with an image layout transition in which case we
  // need an image memory barrier to signal the layout transition. This
  // section handles the follwing 8 transitions leaving us with 32 - 8 = 24
  // possibilities remaining.

  // Read Staging -> Read Buffer
  // Read Staging -> Read Image   (if no layout transition required)
  // Read Buffer  -> Read Staging
  // Read Buffer  -> Read Buffer
  // Read Buffer  -> Read Image   (if no layout transition required)
  // Read Image   -> Read Staging
  // Read Image   -> Read Buffer
  // Read Image   -> Read Image   (if no layout transition required)

  if ((Access::Read == (state.current.view.access & Access::Read)) &&
      (Access::Read == (state.next.view.access & Access::Read)) &&
      !state.requires_image_layout_transition()) {
    return;
  }

  // All transitions after this point require an explicit synchronization of
  // one type or another.

  api::Command::Buffer command_buffer = context_->command().pool.allocate();
  command_buffer.begin();

  // WAR (Write after Read) hazards do not need a memory barrier. Execution
  // barriers are sufficient, unless we are dealing with image layout
  // transitions which do require an image memory barrier to signal the layout
  // transition.  This section handles the following 8 WAR transitions, leaving
  // us with 24 - 8 = 16 remaining possibilities.

  // Read Staging -> Write Buffer
  // Read Staging -> Write Image   (if no layout transition required)
  // Read Buffer  -> Write Staging
  // Read Buffer  -> Write Buffer
  // Read Buffer  -> Write Image   (if no layout transition required)
  // Read Image   -> Write Staging
  // Read Image   -> Write Buffer
  // Read Image   -> Write Image   (if no layout transition required)

  if ((Access::Read == (state.current.view.access & Access::Read)) &&
      // Notice how we include Read-Writes, in addition to Writes, as a
      // Write operation in the condition below as well, as we should.
      (state.next.view.access & Access::Write) &&
      !state.requires_image_layout_transition()) {
    command_buffer.barrier({
      {
        state.current.stage(),
        state.next.stage(),
      },
    });
  }

  // Handle any of the previous 6 RAR or WAR transitions that required a change
  // in image layout, if any.

  // Read Staging -> Read  Image  (if layout transition required)
  // Read Buffer  -> Read  Image  (if layout transition required)
  // Read Image   -> Read  Image  (if layout transition required)
  // Read Staging -> Write Image  (if layout transition required)
  // Read Buffer  -> Write Image  (if layout transition required)
  // Read Image   -> Write Image  (if layout transition required)

  else if (Access::Read == (state.current.view.access & Access::Read)) {
    TORCH_INTERNAL_ASSERT(
        Component::Image == state.next.view.component,
        "Invalid state!  "
        "All RAR or RAW transitions to a non-image must have been handled by now.");

    command_buffer.barrier({
      {
        state.current.stage(),
        state.next.stage(),
      },
      api::Resource::Image::Barrier{
        image().object.handle,
        {
          // Filter out host read and writes.  The spec guarantees a memory dependency
          // and writes
          // https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#synchronization-submission-host-writes
          state.current.access() & ~(VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT),
          state.next.access(),
        },
        {
          state.current.layout(),
          state.next.layout(),
        },
      },
    });
  }

  // Or the remaining 16 RAW or WAW hazards.

  // Write Staging -> Read  Buffer   copy +
  // Write Staging -> Read  Image    image memory barrier
  // Write Staging -> Write Buffer   memory barrier
  // Write Staging -> Write Image    memory barrier
  //
  // Write Buffer  -> Read  Staging
  // Write Buffer  -> Write Staging
  // Write Buffer  -> Read  Buffer
  // Write Buffer  -> Read  Image
  // Write Buffer  -> Write Buffer
  // Write Buffer  -> Write Image
  //
  // Write Image   -> Read  Staging
  // Write Image   -> Write Staging
  // Write Image   -> Read  Buffer
  // Write Image   -> Read  Image
  // Write Image   -> Write Buffer
  // Write Image   -> Write Image

  else {
    TORCH_INTERNAL_ASSERT(
        (state.current.view.access & Access::Write),
        "Invalid state!  "
        "Only RAW or WAW transitions were expected at this point.");

  }

  command_buffer.end();
  command_buffer.submit(context_->gpu().queue);
}

void vTensor::View::verify() const {
  TORCH_INTERNAL_ASSERT(!image_ || (required_ & Component::Image));
  TORCH_INTERNAL_ASSERT(!staging_ || (required_ & Component::Staging));
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
