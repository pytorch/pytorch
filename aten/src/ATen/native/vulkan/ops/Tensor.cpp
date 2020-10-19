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

vTensor::Access::Flags convert(const VkAccessFlags vk_access) {
  vTensor::Access::Flags access = 0u;

  constexpr VkAccessFlags kRead =
      VK_ACCESS_HOST_READ_BIT |
      VK_ACCESS_MEMORY_READ_BIT |
      VK_ACCESS_SHADER_READ_BIT |
      VK_ACCESS_TRANSFER_READ_BIT |
      VK_ACCESS_UNIFORM_READ_BIT;

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

vTensor::Buffer allocate_buffer(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options) {
  TORCH_CHECK(!sizes.empty(), "Invalid Vulkan tensor size!");
  verify(options);

  // Forward function declaration
  bool requires_staging(api::Context*);

  const VkFlags usage = [context]() {
    VkFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    if (requires_staging(context)) {
      usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    return usage;
  }();

  const auto memory = [context]() -> api::Resource::Memory::Descriptor {
    if (requires_staging(context)) {
      return {
        VMA_MEMORY_USAGE_GPU_ONLY,
        0u,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      };
    }

    return {
      VMA_MEMORY_USAGE_UNKNOWN,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };
  }();

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
          memory,
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
          {
            VMA_MEMORY_USAGE_GPU_ONLY,
            0u,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          },
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
          {
            VMA_MEMORY_USAGE_CPU_ONLY,
            0u,
            0u,
          },
        },
      });
}

vTensor::Fence allocate_fence(
    api::Context* const context) {
  return context->resource().pool.fence();
}

enum class Barrier {
  None,
  Exectution,
  Memory,
};

Barrier categorize(
    const VkAccessFlags vk_src_access,
    const VkAccessFlags vk_dst_access) {
  if (0u == vk_src_access) {
    return Barrier::None;
  }

  const vTensor::Access::Flags src_access = convert(vk_src_access);
  const vTensor::Access::Flags dst_access = convert(vk_dst_access);

  if (vTensor::Access::Read == (src_access & vTensor::Access::Read)) {
    if (vTensor::Access::Read == (dst_access & vTensor::Access::Read)) {
      // RAR (Read after Read)
      return Barrier::None;
    }

    // WAR (Write after Read)
    return Barrier::Exectution;
  }

  // RAW (Read after Write), or WAW (Write after Write)
  return Barrier::Memory;
};

Barrier categorize(
    const VkAccessFlags vk_src_access,
    const VkAccessFlags vk_dst_access,
    const VkImageLayout vk_src_layout,
    const VkImageLayout vk_dst_layout) {
  if (vk_src_layout != vk_dst_layout) {
    return Barrier::Memory;
  }

  return categorize(vk_src_access, vk_dst_access);
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

vTensor::Buffer::Object vTensor::buffer(
    const Access::Flags access) & {
  return view_.buffer(access).object;
}

vTensor::Buffer::Object vTensor::buffer(
    api::Command::Buffer& command_buffer) const & {
  return view_.buffer(command_buffer, Access::Read).object;
}

vTensor::Buffer::Object vTensor::buffer(
    api::Command::Buffer& command_buffer,
    const Access::Flags access) & {
  return view_.buffer(command_buffer, access).object;
}

vTensor::Image::Object vTensor::image() const & {
  return view_.image(Access::Read).object;
}

vTensor::Image::Object vTensor::image(
    const Access::Flags access) & {
  return view_.image(access).object;
}

vTensor::Image::Object vTensor::image(
    api::Command::Buffer& command_buffer) const & {
  return view_.image(command_buffer, Access::Read).object;
}

vTensor::Image::Object vTensor::image(
    api::Command::Buffer& command_buffer,
    const Access::Flags access) & {
  return view_.image(command_buffer, access).object;
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
    strides_(sizes.size()),
    options_(options) {
  ops::verify(options);
}

// We typically do not know whether we need a command buffer to service a request
// until we have perfomed a bunch of checks in nested logic, and even then we
// may end up with the always issued state transition optimized away under
// certain conditions, which makes a policy of always allocating a command buffer
// up front, only to end up using it at times, a wasteful approach.  This class
// answers that need.

class vTensor::View::CMD final {
 public:
  CMD(const View&);
  CMD(const View&, api::Command::Buffer&);
  CMD(const CMD&) = delete;
  CMD& operator=(const CMD&) = delete;
  CMD(CMD&&) = delete;
  CMD& operator=(CMD&&) = delete;
  ~CMD() = default;

  typedef api::Resource::Buffer Buffer;
  typedef api::Resource::Image Image;
  typedef api::Resource::Fence Fence;

  void barrier(State::Transition transition);

  void copy_buffer_to_staging(
      State& state,
      const Buffer::Object& buffer,
      Buffer::Object& staging);

  void copy_staging_to_buffer(
      State& state,
      const Buffer::Object& staging,
      Buffer::Object& buffer);

  void copy_buffer_to_image(
      State& state,
      const Buffer::Object& buffer,
      Image::Object& image);

  void copy_image_to_buffer(
      State& state,
      const Image::Object& image,
      Buffer::Object& buffer);

  void submit(Fence fence = {});

 private:
  api::Command::Buffer& command_buffer();

 private:
  const View& view_;

  enum class Type {
    Internal,
    External,
  } type;

  union {
    api::Command::Buffer internal;
    api::Command::Buffer* external;
  } command_buffer_;
};

vTensor::View::CMD::CMD(
    const View& view)
  : view_(view),
    type(Type::Internal),
    command_buffer_{} {
}

vTensor::View::CMD::CMD(
    const View& view,
    api::Command::Buffer& external)
  : view_(view),
    type(Type::External),
    command_buffer_{
      .external = &external,
    } {
}

api::Command::Buffer& vTensor::View::CMD::command_buffer() {
  switch (type) {
    case Type::Internal:
      if (!command_buffer_.internal) {
        command_buffer_.internal = view_.context_->command().pool.allocate();
        command_buffer_.internal.begin();
      }

      return command_buffer_.internal;

    case Type::External:
      return *(command_buffer_.external);

    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown command buffer type!");
      break;
  }
}

void vTensor::View::CMD::barrier(State::Transition transition) {
  // Buffer and Staging are just an alias for the same memory location on UMA.

  if (view_.state_.is_uma()) {
    transition.first.buffer.stage |= transition.first.staging.stage;
    transition.first.buffer.access |= transition.first.staging.access;
    transition.first.staging = {};

    transition.second.buffer.stage |= transition.second.staging.stage;
    transition.second.buffer.access |= transition.second.staging.access;
    transition.second.staging = {};
  }

  // Filter out host dependencies out of source, per Vulkan spec host write ordering guarantees:
  // https://www.khronos.org/registry/vulkan/specs/1.2/html/vkspec.html#synchronization-submission-host-writes

  const auto filter_stage =[](VkPipelineStageFlags& stage) {
    stage &= ~VK_PIPELINE_STAGE_HOST_BIT;
  };

  filter_stage(transition.first.buffer.stage);
  filter_stage(transition.first.staging.stage);

  const auto filter_access =[](VkAccessFlags& access) {
    access &= ~(VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT);
  };

  filter_access(transition.first.buffer.access);
  filter_access(transition.first.staging.access);

  api::Pipeline::Barrier barrier{};

  if (transition.second.staging) {
    const State::Bundle::Buffer from = transition.first.staging;
    const State::Bundle::Buffer to = transition.second.staging;

    const Barrier category = categorize(
        from.access,
        to.access);

    if (Barrier::None != category) {
      barrier.stage.src |= from.stage;
      barrier.stage.dst |= to.stage;

      if (Barrier::Memory == category) {
        barrier.buffers.push_back({
          view_.staging().object,
          {
            from.access,
            to.access,
          },
        });
      }
    }
  }

  if (transition.second.buffer) {
    const State::Bundle::Buffer from = transition.first.buffer;
    const State::Bundle::Buffer to = transition.second.buffer;

    const Barrier category = categorize(
        from.access,
        to.access);

    if (Barrier::None != category) {
      barrier.stage.src |= from.stage;
      barrier.stage.dst |= to.stage;

      if (Barrier::Memory == category) {
        barrier.buffers.push_back({
          view_.buffer().object,
          {
            from.access,
            to.access,
          },
        });
      }
    }
  }

  if (transition.second.image) {
    const State::Bundle::Image from = transition.first.image;
    const State::Bundle::Image to = transition.second.image;

    const Barrier category = categorize(
        from.access,
        to.access,
        from.layout,
        to.layout);

    if (Barrier::None != category) {
      barrier.stage.src |= from.stage;
      barrier.stage.dst |= to.stage;

      if (Barrier::Memory == category) {
        barrier.images.push_back({
          view_.image().object,
          {
            from.access,
            to.access,
          },
          {
            from.layout,
            to.layout,
          },
        });
      }
    }
  }

  // If we are left with anything meaningful, insert a barrier.

  if (barrier) {
    if (0u == barrier.stage.src) {
      barrier.stage.src = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }

    if (0u == barrier.stage.dst) {
      barrier.stage.src = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    command_buffer().barrier(barrier);
  }
}

void vTensor::View::CMD::copy_buffer_to_staging(
    State& state,
    const Buffer::Object& buffer,
    Buffer::Object& staging) {
  if (state.is_clean(Component::Staging) || state.is_uma()) {
    return;
  }

  barrier(
      state.transition({
          // Staging
          {
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,
          },
          // Buffer
          {
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
          },
          // Image
          {},
        }));

  command_buffer().copy(buffer, staging);
}

void vTensor::View::CMD::copy_staging_to_buffer(
    State& state,
    const Buffer::Object& staging,
    Buffer::Object& buffer) {
  if (state.is_clean(Component::Buffer) || state.is_uma()) {
    return;
  }

  barrier(
      state.transition({
          // Staging
          {
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
          },
          // Buffer
          {
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,
          },
          // Image
          {},
        }));

  command_buffer().copy(staging, buffer);
}

void vTensor::View::CMD::copy_buffer_to_image(
    State& state,
    const Buffer::Object& buffer,
    Image::Object& image) {
  if (state.is_clean(Component::Image)) {
    return;
  }

  barrier(
      state.transition({
          // Staging
          {},
          // Buffer
          {
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT,
          },
          // Image
          {
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
          },
        }));

  const struct {
    uint32_t width;
    uint32_t height;
  } block {
  };

  view_.context_->dispatch(
      command_buffer(),
      {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      },
      VK_KERNEL(nchw_to_image),
      {
        8, 8, 1,
      },
      {
        1, 1, 1,
      },
      image,
      buffer,
      // Object lifetime is managed by the resource pool.
      // It is OK not to keep track of the handle.
      view_.context_->resource().pool.uniform(block).object);
}

void vTensor::View::CMD::copy_image_to_buffer(
    State& state,
    const Image::Object& image,
    Buffer::Object& buffer) {
  if (state.is_clean(Component::Buffer)) {
    return;
  }

  barrier(
      state.transition({
          // Staging
          {},
          // Buffer
          {
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
          },
          // Image
          {
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          },
        }));

  const struct {
    uint32_t width;
    uint32_t height;
  } block {
  };

  view_.context_->dispatch(
      command_buffer(),
      {
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      },
      VK_KERNEL(image_to_nchw),
      {
        8, 8, 1,
      },
      {
        1, 1, 1,
      },
      image,
      buffer,
      // Object lifetime is managed by the resource pool.
      // It is OK not to keep track of the handle.
      view_.context_->resource().pool.uniform(block).object);
}

void vTensor::View::CMD::submit(const api::Resource::Fence fence) {
  if ((Type::Internal == type) && command_buffer_.internal) {
    command_buffer_.internal.end();
    command_buffer_.internal.submit(view_.context_->gpu().queue, fence);
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

vTensor::Buffer& vTensor::View::buffer(
    const Access::Flags access) const {
  CMD command_buffer(*this);
  Buffer& buffer = this->buffer(command_buffer, access);
  command_buffer.submit();

  return buffer;
}

vTensor::Buffer& vTensor::View::buffer(
    api::Command::Buffer& command_buffer_,
    const Access::Flags access) const {
  CMD command_buffer(*this, command_buffer_);
  return buffer(command_buffer, access);
}

vTensor::Buffer& vTensor::View::buffer(
    CMD& command_buffer,
    const Access::Flags access) const {
  if ((access & Access::Read) && state_.is_dirty(Component::Buffer)) {
    if (state_.is_clean(Component::Staging)) {
      command_buffer.copy_staging_to_buffer(
          state_,
          staging(command_buffer, Access::Read).object,
          buffer().object);
    }
    else if (state_.is_clean(Component::Image)) {
      command_buffer.copy_image_to_buffer(
          state_,
          image(command_buffer, Access::Read).object,
          buffer().object);
    }
    else {
      TORCH_INTERNAL_ASSERT(
          false,
          "Invalid state!");
    }
  }

  command_buffer.barrier(
      state_.transition({
          // Staging
          {},
          // Buffer
          {
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            [access]() {
              VkAccessFlags vk_access = 0u;

              if (access & Access::Read) {
                vk_access |= VK_ACCESS_SHADER_READ_BIT;
              }

              if (access & Access::Write) {
                vk_access |= VK_ACCESS_SHADER_WRITE_BIT;
              }

              return vk_access;
            }(),
          },
          // Image
          {},
        }));

  if (access & Access::Write) {
    state_.set_dirty(Component::All);
  }

  state_.set_clean(Component::Buffer);

  return buffer();
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

vTensor::Image& vTensor::View::image(
    const Access::Flags access) const {
  CMD command_buffer(*this);
  Image& image = this->image(command_buffer, access);
  command_buffer.submit();

  return image;
}

vTensor::Image& vTensor::View::image(
    api::Command::Buffer& command_buffer_,
    const Access::Flags access) const {
  CMD command_buffer(*this, command_buffer_);
  return image(command_buffer, access);
}

vTensor::Image& vTensor::View::image(
    CMD& command_buffer,
    const Access::Flags access) const {
  if ((access & Access::Read) && state_.is_dirty(Component::Image)) {
    command_buffer.copy_buffer_to_image(
        state_,
        buffer(command_buffer, Access::Read).object,
        image().object);
  }

  command_buffer.barrier(
      state_.transition({
          // Staging
          {},
          // Buffer
          {},
          // Image
          {
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            [access]() {
              VkAccessFlags vk_access = 0u;

              if (access & Access::Read) {
                vk_access |= VK_ACCESS_SHADER_READ_BIT;
              }

              if (access & Access::Write) {
                vk_access |= VK_ACCESS_SHADER_WRITE_BIT;
              }

              return vk_access;
            }(),
            [access]() {
              if (Access::Read == (access & Access::Read)) {
                return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
              }

              return VK_IMAGE_LAYOUT_GENERAL;
            }(),
          },
        }));

  if (access & Access::Write) {
    state_.set_dirty(Component::All);
  }

  state_.set_clean(Component::Image);

  return image();
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

vTensor::Buffer& vTensor::View::staging(const Access::Flags access) const {
  CMD command_buffer(*this);
  Buffer& staging = this->staging(command_buffer, access);
  command_buffer.submit();

  return staging;
}

vTensor::Buffer& vTensor::View::staging(
    CMD& command_buffer,
    const Access::Flags access) const {
  if ((access & Access::Read) && state_.is_dirty(Component::Staging)) {
    command_buffer.copy_buffer_to_staging(
        state_,
        buffer(command_buffer, Access::Read).object,
        staging().object);
  }

  command_buffer.barrier(
      state_.transition({
          // Staging
          {
            VK_PIPELINE_STAGE_HOST_BIT,
            [access]() {
              VkAccessFlags vk_access = 0u;

              if (access & Access::Read) {
                vk_access |= VK_ACCESS_HOST_READ_BIT;
              }

              if (access & Access::Write) {
                vk_access |= VK_ACCESS_HOST_WRITE_BIT;
              }

              return vk_access;
            }(),
          },
          // Buffer
          {},
          // Image
          {},
        }));

  if (access & Access::Write) {
    state_.set_dirty(Component::All);
  }

  state_.set_clean(Component::Staging);

  return staging();
}

vTensor::Memory& vTensor::View::wait() const {
  if (fence_) {
    fence_.wait();
  }

  return staging().memory;
}

vTensor::Fence& vTensor::View::fence() const {
  return (fence_ = allocate_fence(context_));
}

void vTensor::View::verify() const {
  TORCH_INTERNAL_ASSERT(!image_ || state_.is_available(Component::Image));
  TORCH_INTERNAL_ASSERT(!staging_ || state_.is_discrete());
}

vTensor::View::State::State()
  : available_{},
    dirty_{},
    bundle_{} {
}

vTensor::View::State::State(
    api::Context* const context,
    const IntArrayRef sizes)
  : available_{},
    dirty_{},
    bundle_{} {
  available_ |= Component::Buffer;

  if (requires_image(sizes)) {
    available_ |= Component::Image;
  }

  if (requires_staging(context)) {
    available_ |= Component::Staging;
  }
}

vTensor::View::State::Transition
vTensor::View::State::transition(const Bundle bundle) {
  const Bundle from = bundle_;
  Bundle& to = bundle_;

  if (bundle.staging) {
    to.staging = bundle.staging;
  }

  if (bundle.buffer) {
    to.buffer = bundle.buffer;
  }

  if (bundle.image) {
    to.image = bundle.image;
  }

#ifdef DEBUG
  // Forward declaration
  std::ostream& operator<<(
      std::ostream&,
      const View::State::Bundle&);

  std::cout << "From:" << std::endl << from << std::endl;
  std::cout << "To:" << std::endl << to << std::endl;
#endif /* DEBUG */

  return Transition{
    from,
    to,
  };
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

//
// Debug
//

namespace {

// Considering that VkAccessFlags is a weak typedef of a built-in data type, we
// need to introduce a new type to allow overload resolution distinguish between
// the two.

struct Access final {
  VkAccessFlags value;
};

std::ostream& operator<<(
    std::ostream& stream,
    const Access& access) {
  stream << "Access: ";

  if (0u == access.value) {
    return stream << "  0";
  }

  if (access.value & VK_ACCESS_HOST_READ_BIT) {
    stream << "  VK_ACCESS_HOST_READ_BIT";
  }

  if (access.value & VK_ACCESS_HOST_WRITE_BIT) {
    stream << "  VK_ACCESS_HOST_WRITE_BIT";
  }

  if (access.value & VK_ACCESS_MEMORY_READ_BIT) {
    stream << "  VK_ACCESS_MEMORY_READ_BIT";
  }

  if (access.value & VK_ACCESS_MEMORY_WRITE_BIT) {
    stream << "  VK_ACCESS_MEMORY_WRITE_BIT";
  }

  if (access.value & VK_ACCESS_SHADER_READ_BIT) {
    stream << "  VK_ACCESS_SHADER_READ_BIT";
  }

  if (access.value & VK_ACCESS_SHADER_WRITE_BIT) {
    stream << "  VK_ACCESS_SHADER_WRITE_BIT";
  }

  if (access.value & VK_ACCESS_TRANSFER_READ_BIT) {
    stream << "  VK_ACCESS_TRANSFER_READ_BIT";
  }

  if (access.value & VK_ACCESS_TRANSFER_WRITE_BIT) {
    stream << "  VK_ACCESS_TRANSFER_WRITE_BIT";
  }

  return stream;
}

// Considering that VkImageLayout is a weak typedef of a built-in data type,
// we need to introduce a new type to allow overload resolution distinguish
// between the two.

struct Layout final {
  VkImageLayout value;
};

std::ostream& operator<<(
    std::ostream& stream,
    const Layout& layout) {
  stream << "Layout: ";

  switch (layout.value) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
      stream << "  VK_IMAGE_LAYOUT_UNDEFINED";
      break;

    case VK_IMAGE_LAYOUT_GENERAL:
      stream << "  VK_IMAGE_LAYOUT_GENERAL";
      break;

    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      stream << "  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL";
      break;

    default:
      stream << "  Unknown!";
      break;
  };

  return stream;
}

// Considering that VkPipelineStageFlags is a weak typedef of a built-in data
// type, we need to introduce a new type to allow overload resolution distinguish
// between the two.

struct Stage final {
  VkPipelineStageFlags value;
};

std::ostream& operator<<(
    std::ostream& stream,
    const Stage& stage) {
  stream << "Stage: ";

  if (0u == stage.value) {
    return stream << "  0";
  }

  if (stage.value & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
    stream << "  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT";
  }

  if (stage.value & VK_PIPELINE_STAGE_HOST_BIT) {
    stream << "  VK_PIPELINE_STAGE_HOST_BIT";
  }

  if (stage.value & VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) {
    stream << "  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT";
  }

  if (stage.value & VK_PIPELINE_STAGE_TRANSFER_BIT) {
    stream << "  VK_PIPELINE_STAGE_TRANSFER_BIT";
  }

  return stream;
}

} // namespace

std::ostream& operator<<(
    std::ostream& stream,
    const vTensor::View::State::Bundle& bundle) {
  stream << "Staging\n " <<
      Stage{
        bundle.staging.stage,
      } << "\n " <<
      Access{
        bundle.staging.access,
      } << std::endl;

  stream << "Buffer\n " <<
      Stage{
        bundle.buffer.stage,
      } << "\n " <<
      Access{
        bundle.buffer.access,
      } << std::endl;

  stream << "Image\n " <<
      Stage{
        bundle.image.stage,
      } << "\n " <<
      Access{
        bundle.image.access,
      } <<  "\n " <<
      Layout{
        bundle.image.layout,
      } << std::endl;

  return stream;
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
