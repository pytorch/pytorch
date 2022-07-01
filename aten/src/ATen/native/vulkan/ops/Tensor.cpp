#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Tensor.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

uvec3 image_extents(IntArrayRef);
bool requires_image(IntArrayRef);
bool requires_staging(const api::Adapter*);

VkFormat vk_format(const caffe2::TypeMeta dtype) {
  switch (c10::typeMetaToScalarType(dtype)) {
    case kFloat:
    #ifdef USE_VULKAN_FP16_INFERENCE
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    #else
      return VK_FORMAT_R32G32B32A32_SFLOAT;
    #endif /* USE_VULKAN_FP16_INFERENCE */

    default:
      TORCH_CHECK(
          false,
          "Vulkan tensor format not supported!");
  }

  return VK_FORMAT_UNDEFINED;
}

VkExtent3D vk_extent(const uvec3& extent) {
  return {
    extent.data[0u],
    extent.data[1u],
    extent.data[2u],
  };
}

api::MemoryAccessFlags access(
    const VkAccessFlags vk_access) {
  api::MemoryAccessFlags access = 0u;

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
    access |= api::MemoryAccessType::READ;
  }

  if (vk_access & kWrite) {
    access |= api::MemoryAccessType::WRITE;
  }

  return access;
}

VkAccessFlags vk_access(
    const vTensor::Stage::Flags stage,
    const api::MemoryAccessFlags access) {
  VkAccessFlags vk_access = 0u;

  if (access & api::MemoryAccessType::READ) {
    if (stage & vTensor::Stage::Compute) {
      vk_access |= VK_ACCESS_SHADER_READ_BIT;
    }

    if (stage & vTensor::Stage::Host) {
      vk_access |= VK_ACCESS_HOST_READ_BIT;
    }

    if (stage & vTensor::Stage::Transfer) {
      vk_access |= VK_ACCESS_TRANSFER_READ_BIT;
    }
  }

  if (access & api::MemoryAccessType::WRITE) {
    if (stage & vTensor::Stage::Compute) {
      vk_access |= VK_ACCESS_SHADER_WRITE_BIT;
    }

    if (stage & vTensor::Stage::Host) {
      vk_access |= VK_ACCESS_HOST_WRITE_BIT;
    }

    if (stage & vTensor::Stage::Transfer) {
      vk_access |= VK_ACCESS_TRANSFER_WRITE_BIT;
    }
  }

  return vk_access;
}

VkImageLayout vk_layout(
    const vTensor::Stage::Flags stage,
    const api::MemoryAccessFlags access) {
  switch (stage) {
    case vTensor::Stage::Compute:
      switch (access) {
        case api::MemoryAccessType::READ:
          return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        default:
          return VK_IMAGE_LAYOUT_GENERAL;
      } break;

    case vTensor::Stage::Transfer:
      switch (access) {
        case api::MemoryAccessType::READ:
          return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        case api::MemoryAccessType::WRITE:
          return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        default:
          TORCH_INTERNAL_ASSERT(false, "Invalid!");
      } break;

    default:
      TORCH_INTERNAL_ASSERT(false, "Invalid!");
  }

  return VK_IMAGE_LAYOUT_UNDEFINED;
}

VkPipelineStageFlags vk_stage(
    const vTensor::Stage::Flags stage) {
  VkPipelineStageFlags vk_stage = 0u;

  if (stage & vTensor::Stage::Compute) {
    vk_stage |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }

  if (stage & vTensor::Stage::Host) {
    vk_stage |= VK_PIPELINE_STAGE_HOST_BIT;
  }

  if (stage & vTensor::Stage::Transfer) {
    vk_stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;
  }

  return vk_stage;
}

VkDeviceSize buffer_bytes(
    const IntArrayRef sizes,
    const caffe2::TypeMeta dtype) {
  VkDeviceSize size = c10::elementSize(c10::typeMetaToScalarType(dtype));

  if (requires_image(sizes)) {
    const uvec3 extents = image_extents(sizes);
    size *= extents.data[0u] * extents.data[1u] * (4u * extents.data[2u]);
  }
  else {
    size *= c10::multiply_integers(sizes);
  }

  return size;
}

bool requires_image(const IntArrayRef sizes) {
  return (1u <= sizes.size()) && (sizes.size() <= 4u);
}

uvec3 image_extents(const IntArrayRef sizes) {
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

  return {
    safe_downcast<uint32_t>(width),
    safe_downcast<uint32_t>(height),
    safe_downcast<uint32_t>(div_up(depth, INT64_C(4))),
  };
}

bool requires_staging(const api::Adapter* const adapter) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      adapter,
      "Invalid Vulkan adapter!");

  return !adapter->has_unified_memory();
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

  const api::MemoryAccessFlags src_access = access(vk_src_access);
  const api::MemoryAccessFlags dst_access = access(vk_dst_access);

  if ((src_access & api::MemoryAccessType::READ) == src_access) {
    if ((dst_access & api::MemoryAccessType::READ) == dst_access) {
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
  : view_(new View{
      context,
      sizes,
      options,
    }) {
}

api::VulkanBuffer& vTensor::host_buffer(
    api::Command::Buffer& command_buffer,
    const api::MemoryAccessFlags access) & {
  return view_->staging(command_buffer, Stage::Host, access);
}

api::VulkanBuffer::Package vTensor::buffer(
    api::Command::Buffer& command_buffer,
    const Stage::Flags stage) const & {
  return view_->buffer(
      command_buffer,
      stage,
      api::MemoryAccessType::READ).package();
}

api::VulkanBuffer::Package vTensor::buffer(
    api::Command::Buffer& command_buffer,
    const Stage::Flags stage,
    const api::MemoryAccessFlags access) & {
  return view_->buffer(
      command_buffer,
      stage,
      access).package();
}

api::VulkanImage::Package vTensor::image(
    api::Command::Buffer& command_buffer,
    const Stage::Flags stage) const & {
  return view_->image(
      command_buffer,
      stage,
      api::MemoryAccessType::READ).package();
}

api::VulkanImage::Package vTensor::image(
    api::Command::Buffer& command_buffer,
    const Stage::Flags stage,
    const api::MemoryAccessFlags access) & {
  return view_->image(
      command_buffer,
      stage,
      access).package();
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
    state_{},
    // Metadata
    extents_{} {
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
    state_(context->gpu().adapter, sizes),
    // Metadata
    extents_(image_extents(sizes)),
    options_(options),
    sizes_(sizes),
    strides_(sizes.size()) {
  ops::verify(options);
}

vTensor::View::~View() {
  release();
}

void vTensor::View::release() {
  context_->register_image_cleanup(image_);
  context_->register_buffer_cleanup(buffer_);
  if (staging_) {
    context_->register_buffer_cleanup(staging_);
  }
  if (fence_) {
    context_->fences().return_fence(fence_);
  }
}

class vTensor::View::CMD final {
 public:
  CMD(const View&, api::Command::Buffer&);
  CMD(const CMD&) = delete;
  CMD& operator=(const CMD&) = delete;
  CMD(CMD&&) = delete;
  CMD& operator=(CMD&&) = delete;
  ~CMD() = default;

  void barrier(State::Transition transition);

  void copy_buffer_to_staging(
      State& state,
      const api::VulkanBuffer& buffer,
      api::VulkanBuffer& staging);

  void copy_staging_to_buffer(
      State& state,
      const api::VulkanBuffer& staging,
      api::VulkanBuffer& buffer);

  void copy_buffer_to_image(
      State& state,
      const api::VulkanBuffer& buffer,
      api::VulkanImage& image);

  void copy_image_to_buffer(
      State& state,
      const api::VulkanImage& image,
      api::VulkanBuffer& buffer);

  void submit(api::VulkanFence& fence);

 private:
  const View& view_;
  api::Command::Buffer& command_buffer_;
};

vTensor::View::CMD::CMD(
    const View& view,
    api::Command::Buffer& command_buffer)
  : view_(view),
    command_buffer_(command_buffer) {
}

void vTensor::View::CMD::barrier(State::Transition transition) {
  // Buffer and Staging are just an alias for the same memory region on UMA.

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

  api::PipelineBarrier barrier{};

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
        barrier.buffers.push_back(
            api::BufferMemoryBarrier(from.access, to.access, view_.staging()));
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
        barrier.buffers.push_back(
            api::BufferMemoryBarrier(from.access, to.access, view_.buffer()));
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
        TORCH_INTERNAL_ASSERT(
            from.layout == view_.image().layout(),
            "Invalid image layout!");
        api::VulkanImage::Package package = view_.image().package();

        barrier.images.push_back(
            api::ImageMemoryBarrier(
                from.access,
                to.access,
                from.layout,
                to.layout,
                view_.image()));

        view_.image().set_layout(to.layout);
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

    command_buffer_.barrier(barrier);
  }
}

void vTensor::View::CMD::copy_buffer_to_staging(
    State& state,
    const api::VulkanBuffer& buffer,
    api::VulkanBuffer& staging) {
  if (state.is_clean(Component::Staging) || state.is_uma()) {
    return;
  }

  barrier(
      state.transition({
          // Staging
          {
            vk_stage(Stage::Transfer),
            vk_access(Stage::Transfer, api::MemoryAccessType::WRITE),
          },
          // Buffer
          {
            vk_stage(Stage::Transfer),
            vk_access(Stage::Transfer, api::MemoryAccessType::READ),
          },
          // Image
          {},
        }));

  command_buffer_.copy(buffer.package(), staging.package());
}

void vTensor::View::CMD::copy_staging_to_buffer(
    State& state,
    const api::VulkanBuffer& staging,
    api::VulkanBuffer& buffer) {
  if (state.is_clean(Component::Buffer) || state.is_uma()) {
    return;
  }

  barrier(
      state.transition({
          // Staging
          {
            vk_stage(Stage::Transfer),
            vk_access(Stage::Transfer, api::MemoryAccessType::READ),
          },
          // Buffer
          {
            vk_stage(Stage::Transfer),
            vk_access(Stage::Transfer, api::MemoryAccessType::WRITE),
          },
          // Image
          {},
        }));

  command_buffer_.copy(staging.package(), buffer.package());
}

void vTensor::View::CMD::copy_buffer_to_image(
    State& state,
    const api::VulkanBuffer& buffer,
    api::VulkanImage& image) {
  if (state.is_clean(Component::Image)) {
    return;
  }

  api::OpProfiler profiler(command_buffer_, view_.context_->querypool(), "copy_buffer_to_image");

  barrier(
      state.transition({
          // Staging
          {},
          // Buffer
          {
            vk_stage(Stage::Compute),
            vk_access(Stage::Compute, api::MemoryAccessType::READ),
          },
          // Image
          {
            vk_stage(Stage::Compute),
            vk_access(Stage::Compute, api::MemoryAccessType::WRITE),
            vk_layout(Stage::Compute, api::MemoryAccessType::WRITE),
          },
        }));

  const uvec3 extents = view_.extents();
  const uint32_t plane = extents.data[0u] * extents.data[1u];

  const struct Block final {
    uvec3 extents;
    uint32_t block;
    uvec4 offset;
  } block {
    extents,
    4u * plane,
    {
      0u * plane,
      1u * plane,
      2u * plane,
      3u * plane,
    },
  };

  api::UniformParamsBuffer params(view_.context_, block);

  view_.context_->dispatch(
      command_buffer_,
      {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      },
      VK_KERNEL(nchw_to_image),
      extents,
      adaptive_work_group_size(extents),
      image.package(),
      buffer.package(),
      params.buffer().package());
}

void vTensor::View::CMD::copy_image_to_buffer(
    State& state,
    const api::VulkanImage& image,
    api::VulkanBuffer& buffer) {
  if (state.is_clean(Component::Buffer)) {
    return;
  }

  api::OpProfiler profiler(command_buffer_, view_.context_->querypool(), "copy_image_to_buffer");

  barrier(
      state.transition({
          // Staging
          {},
          // Buffer
          {
            vk_stage(Stage::Compute),
            vk_access(Stage::Compute, api::MemoryAccessType::WRITE),
          },
          // Image
          {
            vk_stage(Stage::Compute),
            vk_access(Stage::Compute, api::MemoryAccessType::READ),
            vk_layout(Stage::Compute, api::MemoryAccessType::READ),
          },
        }));

  const uvec3 extents = view_.extents();
  const uint32_t plane = extents.data[0u] * extents.data[1u];

  const struct Block final {
    uvec3 extents;
    uint32_t block;
    uvec4 offset;
  } block {
    extents,
    4u * plane,
    {
      0u * plane,
      1u * plane,
      2u * plane,
      3u * plane,
    },
  };

  api::UniformParamsBuffer params(view_.context_, block);

  view_.context_->dispatch(
      command_buffer_,
      {
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      },
      VK_KERNEL(image_to_nchw),
      view_.extents(),
      adaptive_work_group_size(view_.extents()),
      image.package(),
      buffer.package(),
      params.buffer().package());
}

void vTensor::View::CMD::submit(api::VulkanFence& fence) {
  view_.context_->command().pool.submit(
      view_.context_->gpu().queue,
      command_buffer_,
      fence.get_submit_handle());
}

api::VulkanBuffer& vTensor::View::buffer() const {
  if (!buffer_) {
    api::Adapter* adapter_ptr = context_->adapter_ptr();
    const bool gpu_only = !(adapter_ptr->has_unified_memory());
    buffer_ = adapter_ptr->vma().create_storage_buffer(
        buffer_bytes(sizes(), options().dtype()), gpu_only);
  }

  return buffer_;
}

api::VulkanBuffer& vTensor::View::buffer(
    api::Command::Buffer& command_buffer,
    const Stage::Flags stage,
    const api::MemoryAccessFlags access) const {
  CMD cmd(*this, command_buffer);
  return buffer(cmd, stage, access);
}

api::VulkanBuffer& vTensor::View::buffer(
    CMD& cmd,
    const Stage::Flags stage,
    const api::MemoryAccessFlags access) const {
  if ((access & api::MemoryAccessType::READ) && state_.is_dirty(Component::Buffer)) {
    if (state_.is_clean(Component::Staging)) {
      cmd.copy_staging_to_buffer(
          state_,
          staging(cmd, Stage::Transfer, api::MemoryAccessType::READ),
          buffer());
    }
    else if (state_.is_clean(Component::Image)) {
      cmd.copy_image_to_buffer(
          state_,
          image(cmd, Stage::Compute, api::MemoryAccessType::READ),
          buffer());
    }
    else {
      TORCH_INTERNAL_ASSERT(
          false,
          "Invalid state!");
    }
  }

  cmd.barrier(
      state_.transition({
          // Staging
          {},
          // Buffer
          {
            vk_stage(stage),
            vk_access(stage, access),
          },
          // Image
          {},
        }));

  if (access & api::MemoryAccessType::WRITE) {
    state_.set_dirty(Component::All);
  }

  state_.set_clean(Component::Buffer);

  return buffer();
}

api::VulkanImage& vTensor::View::image() const {
  if (!image_ && state_.is_available(Component::Image)) {
    api::Adapter* adapter_ptr = context_->adapter_ptr();

    api::ImageSampler::Properties sampler_props{
      VK_FILTER_NEAREST,
      VK_SAMPLER_MIPMAP_MODE_NEAREST,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
    };

    VkSampler sampler = adapter_ptr->sampler_cache().retrieve(sampler_props);

    image_ = adapter_ptr->vma().create_image3d_fp(
        vk_extent(extents()), sampler_props, sampler, true);

  }

  return image_;
}

api::VulkanImage& vTensor::View::image(
    api::Command::Buffer& command_buffer,
    const Stage::Flags stage,
    const api::MemoryAccessFlags access) const {
  CMD cmd(*this, command_buffer);
  return image(cmd, stage, access);
}

api::VulkanImage& vTensor::View::image(
    CMD& cmd,
    const Stage::Flags stage,
    const api::MemoryAccessFlags access) const {
  if ((access & api::MemoryAccessType::READ) && state_.is_dirty(Component::Image)) {
    cmd.copy_buffer_to_image(
        state_,
        buffer(cmd, stage, api::MemoryAccessType::READ),
        image());
  }

  cmd.barrier(
      state_.transition({
          // Staging
          {},
          // Buffer
          {},
          // Image
          {
            vk_stage(stage),
            vk_access(stage, access),
            vk_layout(stage, access),
          },
        }));

  if (access & api::MemoryAccessType::WRITE) {
    state_.set_dirty(Component::All);
  }

  state_.set_clean(Component::Image);

  return image();
}

api::VulkanBuffer& vTensor::View::staging() const {
  if (!state_.is_available(Component::Staging)) {
    return buffer();
  }

  if (!staging_) {
    api::Adapter* adapter_ptr = context_->adapter_ptr();
    staging_ = adapter_ptr->vma().create_staging_buffer(
        buffer_bytes(sizes(), options().dtype()));
  }

  return staging_;
}

api::VulkanBuffer& vTensor::View::staging(
    api::Command::Buffer& command_buffer,
    const Stage::Flags stage,
    const api::MemoryAccessFlags access) const {
  CMD cmd(*this, command_buffer);
  api::VulkanBuffer& staging = this->staging(cmd, stage, access);
  cmd.submit(fence(access));

  return staging;
}

api::VulkanBuffer& vTensor::View::staging(
    CMD& cmd,
    const Stage::Flags stage,
    const api::MemoryAccessFlags access) const {
  if ((access & api::MemoryAccessType::READ) && state_.is_dirty(Component::Staging)) {
    cmd.copy_buffer_to_staging(
        state_,
        buffer(cmd, Stage::Transfer, api::MemoryAccessType::READ),
        staging());
  }

  cmd.barrier(
      state_.transition({
          // Staging
          {
            vk_stage(stage),
            vk_access(stage, access),
          },
          // Buffer
          {},
          // Image
          {},
        }));

  if (access & api::MemoryAccessType::WRITE) {
    state_.set_dirty(Component::All);
  }

  state_.set_clean(Component::Staging);

  return staging();
}

api::VulkanFence& vTensor::View::fence(
    const api::MemoryAccessFlags access) const {
  if (!fence_ && access & api::MemoryAccessType::READ) {
    fence_ = context_->fences().get_fence();
  }

  return fence_;
}

void vTensor::View::wait_for_fence() const {
  if (fence_) {
    fence_.wait();
  }
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
    const api::Adapter* const adapter,
    const IntArrayRef sizes)
  : available_{},
    dirty_{},
    bundle_{} {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      adapter,
      "Invalid Vulkan adapter!");

  available_ |= Component::Buffer;

  if (requires_image(sizes)) {
    available_ |= Component::Image;
  }

  if (requires_staging(adapter)) {
    available_ |= Component::Staging;
  }
}

#ifdef VULKAN_TENSOR_DEBUG
std::ostream& operator<<(
    std::ostream&,
    const vTensor::View::State::Bundle&);
#endif /* VULKAN_TENSOR_DEBUG */

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

#ifdef VULKAN_TENSOR_DEBUG
  std::cout << "From:" << std::endl << from << std::endl;
  std::cout << "To:" << std::endl << to << std::endl;
#endif /* VULKAN_TENSOR_DEBUG */

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
      !options.has_memory_format() ||
          (c10::MemoryFormat::Contiguous == options.memory_format_opt()),
      "'memory_format' tensor option is not yet supported under Vulkan!");
}

//
// Debug
//

#ifdef VULKAN_TENSOR_DEBUG

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

struct Image final {
  struct Layout final {
    VkImageLayout value;
  };
};

std::ostream& operator<<(
    std::ostream& stream,
    const Image::Layout& layout) {
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
      Image::Layout{
        bundle.image.layout,
      } << std::endl;

  return stream;
}

#endif /* VULKAN_TENSOR_DEBUG */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
