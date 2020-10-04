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

  // Forward function declaration
  bool requires_staging(api::Context*);

  VkFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  if (requires_staging(context)) {
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
  view_.wait();
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

// Any state in column T can transition to any state in column T + 1 for a
// total of 7 x 6 = 42 possible transitions.  In each scenario, synchronization
// must be handled appropriately.
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

  std::cout << "--------------------\n";
  std::cout << "BEGIN\n";
  std::cout << "--------------------\n\n";

  // If the target view is an image, either:
  //   1) Make sure image memory is allocated if the tensor can be represented
  //      as an image, or
  //   2) Adjust the target view to a buffer if the tensor cannot be represented
  //      as an image.

  if (Component::Image == view.component) {
    if (required_ & Component::Image) {
      // Force a lazy allocation.
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

      inline VkAccessFlags access() const {
        return State::access(view, required_) &
               // Filter out host dependencies out of the transition source,
               // per Vulkan spec guarantee:
               // https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#synchronization-submission-host-writes
              ~(VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT);
      }

      inline VkImageLayout layout() const {
        return image.layout;
      }

      inline void layout(const VkImageLayout layout) {
        image.layout = layout;
      }

      inline VkPipelineStageFlags stage() const {
        return State::stage(view, required_) &
               // Filter out host dependencies out of the transition source,
               // per Vulkan spec guarantee:
               // https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#synchronization-submission-host-writes
               ~VK_PIPELINE_STAGE_HOST_BIT;
      }

      Component::Flags required_;
    } current;

    const struct {
      Active view;

      inline VkAccessFlags access() const {
        return State::access(view, required_);
      }

      inline VkImageLayout layout() const {
        return State::layout(view, required_);
      }

      inline VkPipelineStageFlags stage() const {
        return State::stage(view, required_);
      }

      Component::Flags required_;
    } next;

    inline ~State() {
      current.view = next.view;

      if (Component::Image == next.view.component) {
        current.layout(next.layout());
      }

      std::cout << "--------------------\n";
      std::cout << "EXIT\n";
      std::cout << "--------------------\n\n";
    }

    inline bool requires_image_layout_transition() const {
      return (Component::Image == next.view.component) &&
             (next.layout() != current.layout());
    }

    static VkAccessFlags access(
        const Active view,
        const Component::Flags required) {
      const VkPipelineStageFlags stages = stage(view, required);
      VkAccessFlags access = 0u;

      if (stages & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
        if (view.access & Access::Read) {
          access |= VK_ACCESS_SHADER_READ_BIT;
        }

        if (view.access & Access::Write) {
          access |= VK_ACCESS_SHADER_WRITE_BIT;
        }
      }

      if (stages & VK_PIPELINE_STAGE_HOST_BIT) {
        if (view.access & Access::Read) {
          access |= VK_ACCESS_HOST_READ_BIT;
        }

        if (view.access & Access::Write) {
          access |= VK_ACCESS_HOST_WRITE_BIT;
        }
      }

      if (stages & VK_PIPELINE_STAGE_TRANSFER_BIT) {
        if (view.access & Access::Read) {
          access |= VK_ACCESS_TRANSFER_READ_BIT;
        }

        if (view.access & Access::Write) {
          access |= VK_ACCESS_TRANSFER_WRITE_BIT;
        }
      }

      return access;
    }

    static VkImageLayout layout(
        const Active view,
        const Component::Flags required) {
      TORCH_INTERNAL_ASSERT(
          Component::Image == view.component,
          "The active view on the requested Vulkan tensor is not an image!");

      TORCH_INTERNAL_ASSERT(
          (required & Component::Image),
          "This Vulkan tensor cannot have an image representation!");

      if (view.access & Access::Write) {
        return VK_IMAGE_LAYOUT_GENERAL;
      }

      if (Access::Read == (view.access & Access::Read)) {
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }

      return VK_IMAGE_LAYOUT_UNDEFINED;
    }

    static VkPipelineStageFlags stage(
        const Active view,
        const Component::Flags required) {
      // Legend
      //         = UMA =
      // Image   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
      // Buffer  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
      // Staging VK_PIPELINE_STAGE_HOST_BIT
      //         = Discrete =
      // image   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
      // Buffer  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT
      // Staging VK_PIPELINE_STAGE_HOST_BIT           | VK_PIPELINE_STAGE_TRANSFER_BIT

      VkPipelineStageFlags stages = 0u;

      switch (view.component) {
        case Component::Buffer:
        case Component::Image:
          stages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
          break;

        case Component::Staging:
          stages |= VK_PIPELINE_STAGE_HOST_BIT;
          break;

        default:
          break;
      }

      if ((Component::Image != view.component) &&
          (required & Component::Staging)) {
        stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
      }

      return stages;
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
        required_,
      },
      // Next
      {
        view,
        required_,
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

  // If on UMA, Buffer and Staging are both aliases for the same memory region
  // that is both accessible to host and device.  As long as a queue submission
  // command comes in between the host accessing the memory, and the device
  // accessing the same location, the Vulkan spec guarantees all host writes to
  // be visible to device.  https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#synchronization-submission-host-writes
  // CPU readbacks require a barrier and are intentionally not handled here.
  // Furthermore, on discrete systems, this transition requires a transfer from
  // host to device memory which is also handled later on.

  if (!(required_ & Component::Staging) &&
       (Component::Staging == state.current.view.component) &&
       (Component::Buffer == state.next.view.component)) {
    return;
  }

  // RAR (Read after Read) is not a hazard so no synchronization is required
  // unless we are dealing with an image layout transition in which case we
  // need an image memory barrier to signal the layout transition.  This
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

  // WAR (Write after Read) hazards do not need a memory barrier.  Execution
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
      // Notice how we include read-writes, in addition to writes, as a
      // write operation in the condition below as well, as we should.
      (state.next.view.access & Access::Write) &&
      !state.requires_image_layout_transition()) {
    command_buffer.barrier({
      {
        state.current.stage(),
        state.next.stage(),
      },
    });
  }

  // Handle any of the previous 6 RAR or WAR transitions that indeed do require
  // a change in image layout.

  // Read Staging -> Read  Image  (if layout transition required)
  // Read Buffer  -> Read  Image  (if layout transition required)
  // Read Image   -> Read  Image  (if layout transition required)
  // Read Staging -> Write Image  (if layout transition required)
  // Read Buffer  -> Write Image  (if layout transition required)
  // Read Image   -> Write Image  (if layout transition required)

  else if (Access::Read == (state.current.view.access & Access::Read)) {
    TORCH_INTERNAL_ASSERT(
        state.requires_image_layout_transition(),
        "Invalid state!  "
        "All RAR or RAW transitions to a non-image destination must have been "
        "handled by now.");

    // If dealing with a RAR transition to image that requires a change in layout,
    // we do not have a source pipeline stage or memory access dependency, but
    // if dealing with a WAR, we first need to make sure the read is done prior
    // to overwriting the memory.

    command_buffer.barrier({
      {
        (Access::Read == (state.next.view.access & Access::Read)) ?
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT:
            state.next.stage(),
        state.next.stage(),
      },
      api::Resource::Image::Barrier{
        image().object,
        {
          (Access::Read == (state.next.view.access & Access::Read)) ?
              0u :
              state.next.access(),
        },
        {
          state.current.layout(),
          state.next.layout(),
        },
      },
    });
  }

  // Or the remaining 16 RAW or WAW hazards:

  // Write Staging -> Read  Buffer
  // Write Staging -> Read  Image
  // Write Staging -> Write Buffer
  // Write Staging -> Write Image
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
    // Keep in mind that if we have reached here, we must be coming from
    // a write.

    TORCH_INTERNAL_ASSERT(
        (state.current.view.access & Access::Write),
        "Invalid state!  "
        "Only RAW or WAW transitions were expected at this point.");

    // If dealing with a RAW or WAW staging to buffer / image transition:

    if (Component::Staging == state.current.view.component) {
      TORCH_INTERNAL_ASSERT(
          (Component::Buffer == state.next.view.component) ||
          (Component::Image == state.next.view.component),
          "Invalid state!  "
          "Only transitions to buffer or image out of a staging state are "
          "expected at this point.");

      TORCH_INTERNAL_ASSERT(
          (Component::Buffer != state.next.view.component) ||
          (required_ & Component::Staging),
          "Invalid state!  "
          "UMA transitions of staging to buffer are expected to have been "
          "handled earlier.");

      // Submission guarantees host writes being complete according to
      // https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#synchronization-submission-host-writes

      // [No Synchronization Required Here]

      // If on a discrete system, we need to trigger a staging to buffer copy
      // first.  If on UMA, staging to buffer psudo transitions must have been
      // already handled earlier, and execution should have never reached here,
      // which we guard against with the above assertion. There are only staging
      // to image transitions to worry about on UMA in this particular code path.

      if (required_ & Component::Staging) {
        command_buffer.copy(staging().object, buffer().object);

        // Make sure transfer is complete before the buffer is accessed for any
        // further reads or writes.  If our final stop is a buffer, finalize
        // the memory barrier now.  Otherwise wait a bit to combine this barrier
        // with the image layout transition to batch the calls.

        if (Component::Buffer == state.next.view.component) {
          command_buffer.barrier({
            {
              VK_PIPELINE_STAGE_TRANSFER_BIT,
              state.next.stage(),
            },
            api::Resource::Buffer::Barrier{
              buffer().object,
              {
                VK_ACCESS_TRANSFER_WRITE_BIT,
                state.next.access(),
              },
            },
          });
        }
      }

      // Regardless of whether we are on UMA and managed to opportunistically
      // skip the copy above, or are on discrete and had to perform the copy,
      // if our final destination is an image, we need to pack NHWC to NC4HW.

      if (Component::Image == state.next.view.component) {
        // First off, we need to make sure the image is in proper layout for
        // shader storage writes in case it already is not.  Regardless of
        // whether a layout transition is required or not though, we must make
        // sure the staging to buffer copy above is done prior to packing.

        // command_buffer.barrier({
        //   {
        //     VK_PIPELINE_STAGE_TRANSFER_BIT |
        //         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        //     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        //   },
        //   api::Resource::Image::Barrier{
        //   },
        //   api::Resource::Image::Barrier{
        //     image().object,
        //     {
        //       VK_ACCESS_SHADER_READ_BIT,
        //       VK_ACCESS_SHADER_WRITE_BIT,
        //     },
        //     {
        //       state.current.layout(),
        //       VK_IMAGE_LAYOUT_GENERAL,
        //     },
        //   },
        // });

        // Perform NHWC to NC4HW packing:

        //
        // bind pipeline
        // bind descriptor set
        // dispatch
        //

        // Finally, make sure we transition to the target view and layout.
        // The image layout transition could possibly be skipped if source and
        // destination of this transition have the same layout, but we need
        // the memory barrier portion regardless, considering that we just wrote
        // to the image, and need to make the writes visible to anything that
        // comes after whether it is an image read or an image write.

        command_buffer.barrier({
          {
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            state.next.stage(),
          },
          api::Resource::Image::Barrier{
            image().object,
            {
              VK_ACCESS_SHADER_WRITE_BIT,
              state.next.access(),
            },
            {
              VK_IMAGE_LAYOUT_GENERAL,
              state.next.layout(),
            },
          },
        });
      }
    }

    // If dealing with a RAW or WAW buffer to image / staging transition:

    else if (Component::Buffer == state.current.view.component) {
      // Considering that we are coming from a [buffer] write, we need to make
      // the writes visible to whatever operation comes next, regardless of
      // whether we are going to staging (on UMA or discrete), or image.

      command_buffer.barrier({
        {
          state.current.stage(),
          state.next.stage(),
        },
        api::Resource::Buffer::Barrier{
          buffer().object,
          {
            state.current.access(),
            state.next.access(),
          },
        },
      });

      if (Component::Staging == state.next.view.component) {
        command_buffer.copy(buffer().object, staging().object);

        command_buffer.barrier({
          {
            state.current.stage(),
            state.next.stage(),
          },
          api::Resource::Buffer::Barrier{
            buffer().object,
            {
              state.current.access(),
              state.next.access(),
            },
          },
        });

        if (required_ & Component::Staging) {
        }
      }
    }

    // If dealing with a RAW or WAW image to buffer / staging transition:

    else if (Component::Image == state.current.view.component) {
    }

    // Or did we mess up?

    else {
      TORCH_INTERNAL_ASSERT(
          false,
          "Invalid state! Exectution must never reach here.");
    }
  }

  command_buffer.end();
  command_buffer.submit(context_->gpu().queue);
}

void vTensor::View::wait() {
  if (fence_) {
    fence_.wait();
  }
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

std::ostream& operator<<(
    std::ostream& stream,
    const vTensor::View::Active view) {
  using Access = vTensor::Access;
  using Component = vTensor::View::Component;

  stream << "Component: [";
  switch (view.component) {
    case Component::Unknown:
      stream << "Unknown";
      break;

    case Component::Buffer:
      stream << "Buffer";
      break;

    case Component::Image:
      stream << "Image";
      break;

    case Component::Staging:
      stream << "Staging";
      break;

    default:
      stream << "Unknown";
  }

  stream << "], Access: [";

  if (Access::Read == (view.access & Access::Read)) {
    stream << "Read";
  }
  else if (Access::Write == (view.access & Access::Write)) {
    stream << "Write";
  }
  else if (view.access) {
    stream << "Read | Write";
  }
  else {
    stream << "Unknown";
  }

  return stream << "]";
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
