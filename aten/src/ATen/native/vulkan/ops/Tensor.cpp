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

// We typically do not know whether we need a command buffer to service a request
// until we have actually perfomed a bunch of checks in not so trivial nested logic
// which makes a policy of always allocating a command buffer up front, only to end
// up using it at times, a wasteful approach.  This class abstracts that away.

class CMD final {
 public:
  explicit CMD(api::Context* context);
  CMD(const CMD&) = delete;
  CMD& operator=(const CMD&) = delete;
  CMD(CMD&&) = delete;
  CMD& operator=(CMD&&) = delete;
  ~CMD() = default;

  api::Command::Buffer* operator->();
  void submit(api::Resource::Fence fence = {});

 private:
  api::Context* context_;
  api::Command::Buffer command_buffer_;
};

CMD::CMD(api::Context* const context)
  : context_(context) {
}

api::Command::Buffer* CMD::operator->() {
  if (!command_buffer_) {
    command_buffer_ = context_->command().pool.allocate();
    command_buffer_.begin();
  }

  return &command_buffer_;
}

void CMD::submit(const api::Resource::Fence fence) {
  if (command_buffer_) {
    command_buffer_.end();
    command_buffer_.submit(context_->gpu().queue, fence);
  }
}

vTensor::Access::Flags get_access(const VkAccessFlags vk_access) {
  vTensor::Access::Flags access = 0u;

  constexpr VkAccessFlags kRead =
      VK_ACCESS_HOST_READ_BIT |
      VK_ACCESS_MEMORY_READ_BIT |
      VK_ACCESS_SHADER_READ_BIT |
      VK_ACCESS_TRANSFER_READ_BIT;

  constexpr VkAccessFlags kWrite =
      VK_ACCESS_HOST_WRITE_BIT |
      VK_ACCESS_MEMORY_WRITE_BIT |
      VK_ACCESS_SHADER_WRITE_BIT |
      VK_ACCESS_TRANSFER_WRITE_BIT;

  if (vk_access & kRead) {
    access |= vTensor::Access::Read;
  }

  if (vk_access & kWrite) {
    access |= vTensor::Access::Write;
  }

  return access;
}

} // namespace

vTensor::vTensor(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options)
  : view_(context, sizes, options) {
}

const vTensor* vTensor::host() const {
  view_.staging(Access::Read);
  return this;
}

vTensor* vTensor::host(const Access::Flags access) {
  view_.staging(access);
  return this;
}

vTensor::Buffer::Object vTensor::buffer() const & {
  return view_.buffer(Access::Read).object;
}

vTensor::Buffer::Object vTensor::buffer(const Access::Flags access) & {
  return view_.buffer(access).object;
}

vTensor::Image::Object vTensor::image() const & {
  return view_.image(Access::Read).object;
}

vTensor::Image::Object vTensor::image(const Access::Flags access) & {
  return view_.image(access).object;
}

vTensor::View::View()
    // Resources
  : buffer_{},
    image_{},
    staging_{},
    fence_{},
    // Context
    context_(nullptr),
    // State
    state_{} {
  // verify();
}

vTensor::View::View(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options)
    // Resources
  : buffer_{},
    image_{},
    staging_{},
    fence_{},
    // Context
    context_(context),
    // State
    state_(context, sizes),
    // Metadata
    sizes_(sizes),
    options_(options) {
  // verify();
  ops::verify(options);
}

vTensor::Buffer& vTensor::View::buffer(const Access::Flags access) const {
  CMD command_buffer(context_);

  if ((access & Access::Read) && state_.is_dirty(Component::Buffer)) {
    if (state_.is_clean(Component::Staging)) {
      if (state_.is_discrete()) {
        ;
      }
      else {
        ;
      }
    }
    else if (state_.is_clean(Component::Image)) {
      ;
    }
    else {
      ;
    }
  }

  if (access & Access::Write) {
    state_.set_dirty(Component::All);
  }

  state_.set_clean(Component::Buffer);

  return buffer();
}

vTensor::Image& vTensor::View::image(const Access::Flags access) const {
  CMD command_buffer(context_);

  if ((access & Access::Read) && state_.is_dirty(Component::Image)) {
    if (state_.is_dirty(Component::Buffer) && state_.is_clean(Component::Staging)) {
      ;

      state_.is_clean(Component::Buffer);
    }

    if (state_.is_clean(Component::Buffer)) {
      ;
    }
  }

  if (access & Access::Write) {
    state_.set_dirty(Component::All);
  }

  state_.set_clean(Component::Image);

  return image();
}

vTensor::Buffer& vTensor::View::staging(const Access::Flags access) const {
  CMD command_buffer(context_);

  if ((access & Access::Read) && state_.is_dirty(Component::Staging)) {
  }

  if (access & Access::Write) {
    state_.set_dirty(Component::All);
  }

  state_.set_clean(Component::Staging);

  return staging();
}

vTensor::Memory& vTensor::View::wait() {
  if (fence_) {
    fence_.wait();
  }

  return staging().memory;
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
  if (!image_ && state_.is_available(Component::Image)) {
    image_ = allocate_image(
        context_,
        sizes_,
        options_);
  }

  return image_;
}

vTensor::Buffer& vTensor::View::staging() const {
  if (!state_.is_available(Component::Staging)) {
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

vTensor::View::State::State()
  : available_{},
    dirty_{},
    buffer_{},
    image_{},
    staging_{} {
}

vTensor::View::State::State(
    api::Context* const context,
    const IntArrayRef sizes)
  : available_{},
    dirty_{},
    buffer_{},
    image_{},
    staging_{} {

  available_ |= Component::Buffer;

  if (requires_image(sizes)) {
    available_ |= Component::Image;
  }

  if (requires_staging(context)) {
    available_ |= Component::Staging;
  }
}

// void vTensor::View::transition(const State& state) const {
// }

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

// void vTensor::View::transition(Active view) const {
//   verify();

//   std::cout << "--------------------\n";
//   std::cout << "BEGIN\n";
//   std::cout << "--------------------\n\n";

//   // If the target view is an image, either:
//   //   1) Make sure image memory is allocated if the tensor can be represented
//   //      as an image, or
//   //   2) Adjust the target view to a buffer if the tensor cannot be represented
//   //      as an image.

//   if (Component::Image == view.component) {
//     if (required_ & Component::Image) {
//       // Force a lazy allocation.
//       image();
//     }
//     else {
//       // Adjust target view to buffer.
//       view.component = Component::Buffer;
//     }
//   }

//   // Always make sure to update the active view and image layout, if necessary,
//   // regardless of the codepath taken.  Keep in mind that we are simply updating
//   // the state machine here and not issuing any synchronization commands which
//   // is what the rest of the logic in this function takes care of if need be.

//   struct State final {
//     struct {
//       Active& view;
//       Image::Object& image;

//       inline VkAccessFlags access() const {
//         return State::access(view, required_) &
//                // Filter out host dependencies out of the transition source,
//                // per Vulkan spec guarantee:
//                // https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#synchronization-submission-host-writes
//               ~(VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT);
//       }

//       inline VkImageLayout layout() const {
//         return image.layout;
//       }

//       inline void layout(const VkImageLayout layout) {
//         image.layout = layout;
//       }

//       inline VkPipelineStageFlags stage() const {
//         return State::stage(view, required_) &
//                // Filter out host dependencies out of the transition source,
//                // per Vulkan spec guarantee:
//                // https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#synchronization-submission-host-writes
//                ~VK_PIPELINE_STAGE_HOST_BIT;
//       }

//       Component::Flags required_;
//     } current;

//     const struct {
//       Active view;

//       inline VkAccessFlags access() const {
//         return State::access(view, required_);
//       }

//       inline VkImageLayout layout() const {
//         return State::layout(view, required_);
//       }

//       inline VkPipelineStageFlags stage() const {
//         return State::stage(view, required_);
//       }

//       Component::Flags required_;
//     } next;

//     inline ~State() {
//       current.view = next.view;

//       if (Component::Image == next.view.component) {
//         current.layout(next.layout());
//       }

//       std::cout << "--------------------\n";
//       std::cout << "EXIT\n";
//       std::cout << "--------------------\n\n";
//     }

//     inline bool requires_image_layout_transition() const {
//       return (Component::Image == next.view.component) &&
//              (next.layout() != current.layout());
//     }

//     static VkAccessFlags access(
//         const Active view,
//         const Component::Flags required) {
//       const VkPipelineStageFlags stages = stage(view, required);
//       VkAccessFlags access = 0u;

//       if (stages & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
//         if (view.access & Access::Read) {
//           access |= VK_ACCESS_SHADER_READ_BIT;
//         }

//         if (view.access & Access::Write) {
//           access |= VK_ACCESS_SHADER_WRITE_BIT;
//         }
//       }

//       if (stages & VK_PIPELINE_STAGE_HOST_BIT) {
//         if (view.access & Access::Read) {
//           access |= VK_ACCESS_HOST_READ_BIT;
//         }

//         if (view.access & Access::Write) {
//           access |= VK_ACCESS_HOST_WRITE_BIT;
//         }
//       }

//       if (stages & VK_PIPELINE_STAGE_TRANSFER_BIT) {
//         if (view.access & Access::Read) {
//           access |= VK_ACCESS_TRANSFER_READ_BIT;
//         }

//         if (view.access & Access::Write) {
//           access |= VK_ACCESS_TRANSFER_WRITE_BIT;
//         }
//       }

//       return access;
//     }

//     static VkImageLayout layout(
//         const Active view,
//         const Component::Flags required) {
//       TORCH_INTERNAL_ASSERT(
//           Component::Image == view.component,
//           "The active view on the requested Vulkan tensor is not an image!");

//       TORCH_INTERNAL_ASSERT(
//           (required & Component::Image),
//           "This Vulkan tensor cannot have an image representation!");

//       if (view.access & Access::Write) {
//         return VK_IMAGE_LAYOUT_GENERAL;
//       }

//       if (Access::Read == (view.access & Access::Read)) {
//         return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
//       }

//       return VK_IMAGE_LAYOUT_UNDEFINED;
//     }

//     static VkPipelineStageFlags stage(
//         const Active view,
//         const Component::Flags required) {
//       // Legend
//       //         = UMA =
//       // Image   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
//       // Buffer  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
//       // Staging VK_PIPELINE_STAGE_HOST_BIT
//       //         = Discrete =
//       // image   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
//       // Buffer  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT
//       // Staging VK_PIPELINE_STAGE_HOST_BIT           | VK_PIPELINE_STAGE_TRANSFER_BIT

//       VkPipelineStageFlags stages = 0u;

//       switch (view.component) {
//         case Component::Buffer:
//         case Component::Image:
//           stages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
//           break;

//         case Component::Staging:
//           stages |= VK_PIPELINE_STAGE_HOST_BIT;
//           break;

//         default:
//           break;
//       }

//       if ((Component::Image != view.component) &&
//           (required & Component::Staging)) {
//         stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
//       }

//       return stages;
//     }
//   } state {
//       // Current
//       {
//         active_,
//         // Not using the accessor to prevent an unnecessary lazy memory
//         // allocation in cases where either
//         //   1) The tensor cannot be represented as an image, or
//         //   2) The requested target view is not an image.
//         image_.object,
//         required_,
//       },
//       // Next
//       {
//         view,
//         required_,
//       },
//   };

//   std::cout << "--------------------\n";
//   std::cout << "Current - " << state.current.view << std::endl;
//   std::cout << "Next - " << state.next.view << std::endl;
//   std::cout << "--------------------\n\n";

//   // A transition to an unknown state is an invalid transition.  Make sure we
//   // are never going to find ourselves in that boat.

//   TORCH_INTERNAL_ASSERT(
//       Component::Unknown != state.next.view.component,
//       "Invalid transition!");

//   // If dealing with a transition from an initial Unknown state, update the
//   // active view and return.  No synchronization is reuiqred in such scenarios.
//   // This takes care of 6 states originating from an Unknown state leaving us
//   // with 42 - 6 = 36 remaining transitions.

//   if (Component::Unknown == state.current.view.component) {
//     return;
//   }

//   // Memory availability and visibility operations on host is handled through
//   // map() and unmap() if not dealing with coherent host memory.  Other than
//   // that, host to host dependencies require no device-side synchronization.
//   // This is regardless of whether we are dealing with UMA or discrete systems.
//   // This section handles the following 4 transitions which leaves us with
//   // 36 - 4 = 32 possible transitions remaining.

//   // Read  Staging -> Read  Staging
//   // Read  Staging -> Write Staging
//   // Write Staging -> Read  Staging
//   // Write Staging -> Write Staging

//   if ((Component::Staging == state.current.view.component) &&
//       (Component::Staging == state.next.view.component)) {
//     return;
//   }

//   // If on UMA, Buffer and Staging are both aliases for the same memory region
//   // that is both accessible to host and device.  As long as a queue submission
//   // command comes in between the host accessing the memory, and the device
//   // accessing the same location, the Vulkan spec guarantees all host writes to
//   // be visible to device.  https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#synchronization-submission-host-writes
//   // CPU readbacks require a barrier and are intentionally not handled here.
//   // Furthermore, on discrete systems, this transition requires a transfer from
//   // host to device memory which is also handled later on.

//   if (!(required_ & Component::Staging) &&
//        (Component::Staging == state.current.view.component) &&
//        (Component::Buffer == state.next.view.component)) {
//     return;
//   }

//   // RAR (Read after Read) is not a hazard so no synchronization is required
//   // unless we are dealing with an image layout transition in which case we
//   // need an image memory barrier to signal the layout transition.  This
//   // section handles the follwing 8 transitions leaving us with 32 - 8 = 24
//   // possibilities remaining.

//   // Read Staging -> Read Buffer
//   // Read Staging -> Read Image   (if no layout transition required)
//   // Read Buffer  -> Read Staging
//   // Read Buffer  -> Read Buffer
//   // Read Buffer  -> Read Image   (if no layout transition required)
//   // Read Image   -> Read Staging
//   // Read Image   -> Read Buffer
//   // Read Image   -> Read Image   (if no layout transition required)

//   if ((Access::Read == (state.current.view.access & Access::Read)) &&
//       (Access::Read == (state.next.view.access & Access::Read)) &&
//       !state.requires_image_layout_transition()) {
//     return;
//   }

//   // All transitions after this point require an explicit synchronization of
//   // one type or another.

//   api::Command::Buffer command_buffer = context_->command().pool.allocate();
//   command_buffer.begin();

//   // WAR (Write after Read) hazards do not need a memory barrier.  Execution
//   // barriers are sufficient, unless we are dealing with image layout
//   // transitions which do require an image memory barrier to signal the layout
//   // transition.  This section handles the following 8 WAR transitions, leaving
//   // us with 24 - 8 = 16 remaining possibilities.

//   // Read Staging -> Write Buffer
//   // Read Staging -> Write Image   (if no layout transition required)
//   // Read Buffer  -> Write Staging
//   // Read Buffer  -> Write Buffer
//   // Read Buffer  -> Write Image   (if no layout transition required)
//   // Read Image   -> Write Staging
//   // Read Image   -> Write Buffer
//   // Read Image   -> Write Image   (if no layout transition required)

//   if ((Access::Read == (state.current.view.access & Access::Read)) &&
//       // Notice how we include read-writes, in addition to writes, as a
//       // write operation in the condition below as well, as we should.
//       (state.next.view.access & Access::Write) &&
//       !state.requires_image_layout_transition()) {
//     command_buffer.barrier({
//         // Stage
//         {
//           state.current.stage(),
//           state.next.stage(),
//         },
//         // Buffer
//         {},
//         // Image
//         {},
//       });
//   }

//   // Handle any of the previous 6 RAR or WAR transitions that indeed do require
//   // a change in image layout.

//   // Read Staging -> Read  Image  (if layout transition required)
//   // Read Buffer  -> Read  Image  (if layout transition required)
//   // Read Image   -> Read  Image  (if layout transition required)
//   // Read Staging -> Write Image  (if layout transition required)
//   // Read Buffer  -> Write Image  (if layout transition required)
//   // Read Image   -> Write Image  (if layout transition required)

//   else if (Access::Read == (state.current.view.access & Access::Read)) {
//     TORCH_INTERNAL_ASSERT(
//         state.requires_image_layout_transition(),
//         "Invalid state!  "
//         "All RAR or RAW transitions to a non-image destination must have been "
//         "handled by now.");

//     // If dealing with a RAR transition to image that requires a change in layout,
//     // we do not have a source pipeline stage or memory access dependency, but
//     // if dealing with a WAR, we first need to make sure the read is done prior
//     // to overwriting the memory.

//     command_buffer.barrier({
//         // Stage
//         {
//           (Access::Read == (state.next.view.access & Access::Read)) ?
//               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT:
//               state.current.stage(),
//           state.next.stage(),
//         },
//         // Buffer
//         {},
//         // Image
//         {
//           {
//             image().object,
//             {
//               (Access::Read == (state.next.view.access & Access::Read)) ?
//                   0u :
//                   state.current.access(),
//               state.next.access(),
//             },
//             {
//               state.current.layout(),
//               state.next.layout(),
//             },
//           },
//         },
//       });
//   }

//   // Or the remaining 16 RAW or WAW hazards:

//   // Write Staging -> Read  Buffer
//   // Write Staging -> Write Buffer
//   // Write Staging -> Read  Image
//   // Write Staging -> Write Image
//   //
//   // Write Buffer  -> Read  Staging
//   // Write Buffer  -> Write Staging
//   // Write Buffer  -> Read  Buffer
//   // Write Buffer  -> Write Buffer
//   // Write Buffer  -> Read  Image
//   // Write Buffer  -> Write Image
//   //
//   // Write Image   -> Read  Staging
//   // Write Image   -> Write Staging
//   // Write Image   -> Read  Buffer
//   // Write Image   -> Write Buffer
//   // Write Image   -> Read  Image
//   // Write Image   -> Write Image

//   else {
//     // Keep in mind that if we have reached here, we must be coming from
//     // a write.  Anything else violates this expectation.

//     TORCH_INTERNAL_ASSERT(
//         (state.current.view.access & Access::Write),
//         "Invalid state!  "
//         "Only RAW or WAW transitions were expected at this point.");

//     // If dealing with a RAW or WAW staging to buffer / image transition:
//     //
//     // Write Staging -> Read  Buffer
//     // Write Staging -> Write Buffer
//     // Write Staging -> Read  Image
//     // Write Staging -> Write Image

//     if (Component::Staging == state.current.view.component) {
//       // Clearly only expecting staging to buffer or staging to image transitions.
//       // Staging to staging transitions are expected to have been already handled.

//       TORCH_INTERNAL_ASSERT(
//           (Component::Buffer == state.next.view.component) ||
//           (Component::Image == state.next.view.component),
//           "Invalid state!  "
//           "Only transitions to buffer or image out of a staging state are "
//           "expected at this point.  Staging to staging transitions are "
//           "expected to have been handled previously.");

//       // And even then, staging to buffer transitions must have been already
//       // handled on UMA.

//       TORCH_INTERNAL_ASSERT(
//           (Component::Buffer != state.next.view.component) ||
//           (required_ & Component::Staging),
//           "Invalid state!  "
//           "UMA transitions of staging to buffer are expected to have been "
//           "handled earlier.");

//       // Submission guarantees host writes being complete according to
//       // https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#synchronization-submission-host-writes
//       // Consequently:

//       // [No Synchronization Required Here]

//       // If on a discrete system, we need to trigger a staging to buffer copy
//       // first.  If on UMA, staging to buffer psudo transitions must have been
//       // already handled earlier, and execution should never reach here, which
//       // we guard against with the above assertion. There are only staging to
//       // image transitions to worry about on UMA in this particular code path.

//       if (required_ & Component::Staging) {
//         // Trigger a copy on discrete.

//         command_buffer.copy(staging().object, buffer().object);

//         // Make sure transfer is complete before the buffer is accessed for any
//         // further reads or writes.  If our final stop is a buffer, finalize
//         // the memory barrier now.  Otherwise wait a bit longer to combine this
//         // barrier with the image layout transition in an effort to batch the
//         // calls.

//         if (Component::Buffer == state.next.view.component) {
//           command_buffer.barrier({
//               // Stage
//               {
//                 VK_PIPELINE_STAGE_TRANSFER_BIT,
//                 state.next.stage(),
//               },
//               // Buffer
//               {
//                 {
//                   buffer().object,
//                   {
//                     VK_ACCESS_TRANSFER_WRITE_BIT,
//                     state.next.access(),
//                   },
//                 },
//               },
//               // Image
//               {},
//             });
//         }
//       }

//       // Regardless of whether we are on UMA and managed to opportunistically
//       // skip the copy above, or are on discrete and had to perform the copy,
//       // if our final destination is an image, we need to pack NHWC to NC4HW.

//       if (Component::Image == state.next.view.component) {
//         // First off, in case we triggered one, we must make sure the staging to
//         // buffer copy above is done prior to initiating an NHWC to NC4HW packing.
//         // Orthogonollay, we also need to make sure the image is in proper layout
//         // for shader storage writes in case it is not already.  If either of
//         // these scenarios are the case, we need a pre packing barrier.

//         if ((required_ & Component::Staging) ||
//             (VK_IMAGE_LAYOUT_GENERAL != state.current.layout())) {
//           api::Pipeline::Barrier barrier{};

//           // If we are on a discrete system, we must have had triggered a transfer
//           // by this point and need to insert a dependency on that.

//           if (required_ & Component::Staging) {
//             barrier.stage.src |= VK_PIPELINE_STAGE_TRANSFER_BIT;
//             barrier.stage.dst |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

//             barrier.buffers.push_back({
//               buffer().object,
//               {
//                 VK_ACCESS_TRANSFER_WRITE_BIT,
//                 VK_ACCESS_SHADER_READ_BIT,
//               },
//             });
//           }

//           if (VK_IMAGE_LAYOUT_GENERAL != state.current.layout()) {
//             barrier.stage.src |= VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
//             barrier.stage.dst |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

//             barrier.images.push_back({
//               image().object,
//               {
//                 0u,
//                 VK_ACCESS_SHADER_WRITE_BIT,
//               },
//               {
//                 state.current.layout(),
//                 VK_IMAGE_LAYOUT_GENERAL,
//               },
//             });
//           }

//           command_buffer.barrier(barrier);
//         }

//         // Perform NHWC to NC4HW packing:
//         // context_->dispatch();

//         // Finally, make sure we transition to the target view and layout
//         // considering what we have dealt with so far is a transition to an
//         // intermediary state required to perform the packing.  With that said,
//         // the image layout transition could possibly be skipped if source and
//         // destination of this transition have the same layout, but we need
//         // the memory barrier portion regardless, considering that we just wrote
//         // to the image, and need to make the writes visible to anything that
//         // comes after whether it is an image read or an image write.

//         command_buffer.barrier({
//             // Stage
//             {
//               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
//               state.next.stage(),
//             },
//             // Buffer
//             {},
//             // Image
//             {
//               {
//                 image().object,
//                 {
//                   VK_ACCESS_SHADER_WRITE_BIT,
//                   state.next.access(),
//                 },
//                 {
//                   VK_IMAGE_LAYOUT_GENERAL,
//                   state.next.layout(),
//                 },
//               },
//             },
//           });
//       }
//     }

//     // If dealing with a RAW or WAW buffer to image / staging transition:
//     //
//     // Write Buffer -> Read  Staging
//     // Write Buffer -> Write Staging
//     // Write Buffer -> Read  Buffer
//     // Write Buffer -> Write Buffer
//     // Write Buffer -> Read  Image
//     // Write Buffer -> Write Image

//     else if (Component::Buffer == state.current.view.component) {
//       // This code only handles transitions out of buffer to buffer, image, or
//       // staging.  Trnasitions to an unknown state are invalid.

//       TORCH_INTERNAL_ASSERT(
//           (Component::Buffer == state.next.view.component) ||
//           (Component::Image == state.next.view.component) ||
//           (Component::Staging == state.next.view.component),
//           "Invalid state!  "
//           "Only transitions to buffer, image, or staging out of a buffer state "
//           "are expected at this point.");

//       // If we are transitioning to staging on UMA, or buffer on UMA or discrete,
//       // this is a one hop transition.

//       if ((!(required_ & Component::Staging) &&
//             (Component::Staging == state.next.view.component)) ||
//           (Component::Buffer == state.next.view.component)) {
//         command_buffer.barrier({
//             // Stage
//             {
//               state.current.stage(),
//               state.next.stage(),
//             },
//             // Buffer
//             {
//               {
//                 buffer().object,
//                 {
//                   state.current.access(),
//                   state.next.access(),
//                 },
//               },
//             },
//             // Image
//             {},
//           });
//       }

//       // Otherwise, we need to go through the trio of pre-op barrier, op, post-op
//       // barrier where op is either a transfer or packing operation depending on
//       // the destination.

//       else {
//         // No buffer to buffer transitions at this point.

//         TORCH_INTERNAL_ASSERT(
//           (Component::Image == state.next.view.component) ||
//           (Component::Staging == state.next.view.component),
//           "Invalid state!  "
//           "Only transitions to image, or staging out of a buffer state are "
//           "expected at this point.");

//         // The commonalities in the pre-op barrier:

//         api::Pipeline::Barrier barrier{
//           // Stage
//           {
//             state.current.stage(),
//             // To be filled subsequently.
//             0u,
//           },
//           // Buffer
//           {
//             {
//               buffer().object,
//               {
//                 state.current.access(),
//                 // To be filled subsequently.
//                 0u,
//               },
//             },
//           },
//           // Image
//           {},
//         };

//         // Write Buffer -> Read  Staging (discrete)
//         // Write Buffer -> Write Staging (discrete)

//         if (Component::Staging == state.next.view.component) {
//           TORCH_INTERNAL_ASSERT(
//               (required_ & Component::Staging),
//               "Invalid state! "
//               "UMA transitions of buffers to staging are expected to have been "
//               "handled earlier.");

//           // Make sure all writes (remember we are on a RAW or WAW path) on the
//           // source buffer (remember we are on a buffer -> staging path) are done
//           // prior to initiating a buffer to staging transfer.

//           barrier.stage.dst |= VK_PIPELINE_STAGE_TRANSFER_BIT;
//           barrier.buffers.back().memory.dst |= VK_ACCESS_TRANSFER_READ_BIT;
//           command_buffer.barrier(barrier);

//           // Issue the transfer.
//           command_buffer.copy(buffer().object, staging().object);

//           // Make sure transfer writes are made available, and visible to the
//           // requested destination view.

//           command_buffer.barrier({
//               // Stage
//               {
//                 VK_PIPELINE_STAGE_TRANSFER_BIT,
//                 state.next.stage(),
//               },
//               // Buffer
//               {
//                 {
//                   staging().object,
//                   {
//                     VK_ACCESS_TRANSFER_WRITE_BIT,
//                     state.next.access(),
//                   },
//                 },
//               },
//               // Image
//               {},
//             });
//         }

//         // Write Buffer -> Read  Image
//         // Write Buffer -> Write Image

//         else if (Component::Image == state.next.view.component) {
//           // Make sure all writes (remember we are on a RAW or WAW path) on the
//           // source buffer (remember we are on a buffer -> image path) are done
//           // prior to initiating a buffer to image packing.  If a layout
//           // transition is required, handle that here as well.

//           barrier.stage.dst |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
//           barrier.buffers.back().memory.dst |= VK_ACCESS_SHADER_READ_BIT;

//           if (VK_IMAGE_LAYOUT_GENERAL != state.current.layout()) {
//             barrier.stage.src |= VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
//             barrier.stage.dst |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
//             barrier.images.push_back({
//               image().object,
//               {
//                 0u,
//                 VK_ACCESS_SHADER_WRITE_BIT,
//               },
//               {
//                 state.current.layout(),
//                 VK_IMAGE_LAYOUT_GENERAL,
//               },
//             });
//           }

//           command_buffer.barrier(barrier);

//           // Perform NHWC to NC4HW packing:
//           // context_->dispatch();

//           // Make sure transfer writes are made available, and visible to the
//           // requested destination view.

//           command_buffer.barrier({
//               // Stage
//               {
//                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
//                 state.next.stage(),
//               },
//               // Buffer
//               {},
//               // Image
//               {
//                 {
//                   image().object,
//                   {
//                     VK_ACCESS_SHADER_WRITE_BIT,
//                     state.next.access(),
//                   },
//                   {
//                     VK_IMAGE_LAYOUT_GENERAL,
//                     state.next.layout(),
//                   },
//                 },
//               },
//             });
//         }

//         // Or did we mess up?

//         else {
//           TORCH_INTERNAL_ASSERT(
//               false,
//               "Invalid state! Exectution must have never reached here.");
//         }
//       }
//     }

//     // Finally, at long last, if dealing with a RAW or WAW image to buffer /
//     // staging transition:
//     //
//     // Write Image -> Read  Staging
//     // Write Image -> Write Staging
//     // Write Image -> Read  Buffer
//     // Write Image -> Write Buffer
//     // Write Image -> Read  Image
//     // Write Image -> Write Image

//     else if (Component::Image == state.current.view.component) {
//       // This code only handles transitions out of image, to buffer, image, or
//       // staging.  Trnasitions to an unknown state are invalid.

//       TORCH_INTERNAL_ASSERT(
//           (Component::Buffer == state.next.view.component) ||
//           (Component::Image == state.next.view.component) ||
//           (Component::Staging == state.next.view.component),
//           "Invalid state!  "
//           "Only transitions to buffer, image, or staging out of an image state "
//           "are expected at this point.");

//       // If we are transitioning to an image view, this is a one hop transition.
//       //
//       // Write Image -> Read  Image
//       // Write Image -> Write Image

//       if (Component::Image == state.current.view.component) {
//         // Regardless of whether we need a layout transition, a RAW or WAW
//         // requires a memory barrier.

//         command_buffer.barrier({
//               // Stage
//               {
//                 state.current.stage(),
//                 state.next.stage(),
//               },
//               // Buffer
//               {},
//               // Image
//               {
//                 {
//                   image().object,
//                   {
//                     state.current.access(),
//                     state.next.access(),
//                   },
//                   {
//                     state.current.layout(),
//                     state.next.layout(),
//                   },
//                 },
//               },
//             });
//       }

//       // Otherwise, we need to go through the trio of pre-op barrier, op, post-op
//       // barrier where op is an unpacking operation depending.
//       //
//       // Write Image -> Read  Staging
//       // Write Image -> Write Staging
//       // Write Image -> Read  Buffer
//       // Write Image -> Write Buffer

//       else {
//         // No image to image transitions at this point.

//         TORCH_INTERNAL_ASSERT(
//           (Component::Buffer == state.next.view.component) ||
//           (Component::Staging == state.next.view.component),
//           "Invalid state!  "
//           "Only transitions to buffer, or staging out of an image state are "
//           "expected at this point.");

//         // Since we are on a RAW or WAW path, we must first make sure all pending
//         // image writes are flushed before unpacking.

//         command_buffer.barrier({
//               // Stage
//               {
//                 state.current.stage(),
//                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
//               },
//               // Buffer
//               {},
//               // Image
//               {
//                 {
//                   image().object,
//                   {
//                     state.current.access(),
//                     VK_ACCESS_SHADER_READ_BIT,
//                   },
//                   {
//                     state.current.layout(),
//                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
//                   },
//                 },
//               },
//             });

//         // Perform NC4HW to NHWC unpacking:
//         // context_->dispatch();
//       }
//     }

//     // Or did we mess up?

//     else {
//       TORCH_INTERNAL_ASSERT(
//           false,
//           "Invalid state! Exectution must have never reached here.");
//     }
//   }

//   command_buffer.end();
//   command_buffer.submit(context_->gpu().queue);
// }

// void vTensor::View::verify() const {
//   TORCH_INTERNAL_ASSERT(!image_ || (required_ & Component::Image));
//   TORCH_INTERNAL_ASSERT(!staging_ || (required_ & Component::Staging));
// }

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

// std::ostream& operator<<(
//     std::ostream& stream,
//     const vTensor::View::Active view) {
//   using Access = vTensor::Access;
//   using Component = vTensor::View::Component;

//   stream << "Component: [";
//   switch (view.component) {
//     case Component::Unknown:
//       stream << "Unknown";
//       break;

//     case Component::Buffer:
//       stream << "Buffer";
//       break;

//     case Component::Image:
//       stream << "Image";
//       break;

//     case Component::Staging:
//       stream << "Staging";
//       break;

//     default:
//       stream << "Unknown";
//   }

//   stream << "], Access: [";

//   if (Access::Read == (view.access & Access::Read)) {
//     stream << "Read";
//   }
//   else if (Access::Write == (view.access & Access::Write)) {
//     stream << "Write";
//   }
//   else if (view.access) {
//     stream << "Read | Write";
//   }
//   else {
//     stream << "Unknown";
//   }

//   return stream << "]";
// }

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
