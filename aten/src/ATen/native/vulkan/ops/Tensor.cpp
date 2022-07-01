#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Tensor.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

api::utils::uvec3 image_extents(IntArrayRef);
bool requires_image(IntArrayRef);

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

VkExtent3D vk_extent(const api::utils::uvec3& extent) {
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
    const api::PipelineStageFlags stage,
    const api::MemoryAccessFlags access) {
  VkAccessFlags vk_access = 0u;

  if (access & api::MemoryAccessType::READ) {
    if (stage & api::PipelineStage::Compute) {
      vk_access |= VK_ACCESS_SHADER_READ_BIT;
    }

    if (stage & api::PipelineStage::Host) {
      vk_access |= VK_ACCESS_HOST_READ_BIT;
    }

    if (stage & api::PipelineStage::Transfer) {
      vk_access |= VK_ACCESS_TRANSFER_READ_BIT;
    }
  }

  if (access & api::MemoryAccessType::WRITE) {
    if (stage & api::PipelineStage::Compute) {
      vk_access |= VK_ACCESS_SHADER_WRITE_BIT;
    }

    if (stage & api::PipelineStage::Host) {
      vk_access |= VK_ACCESS_HOST_WRITE_BIT;
    }

    if (stage & api::PipelineStage::Transfer) {
      vk_access |= VK_ACCESS_TRANSFER_WRITE_BIT;
    }
  }

  return vk_access;
}

VkImageLayout vk_layout(
    const api::PipelineStageFlags stage,
    const api::MemoryAccessFlags access) {
  switch (stage) {

    case api::PipelineStage::Compute:
      switch (access) {
        case api::MemoryAccessType::READ:
          return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        default:
          return VK_IMAGE_LAYOUT_GENERAL;
      } break;

    case api::PipelineStage::Transfer:
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
    const api::PipelineStageFlags stage) {
  VkPipelineStageFlags vk_stage = 0u;

  if (stage & api::PipelineStage::Compute) {
    vk_stage |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  }

  if (stage & api::PipelineStage::Host) {
    vk_stage |= VK_PIPELINE_STAGE_HOST_BIT;
  }

  if (stage & api::PipelineStage::Transfer) {
    vk_stage |= VK_PIPELINE_STAGE_TRANSFER_BIT;
  }

  return vk_stage;
}

api::utils::uvec3 image_extents(const IntArrayRef sizes) {
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
    api::utils::safe_downcast<uint32_t>(width),
    api::utils::safe_downcast<uint32_t>(height),
    api::utils::safe_downcast<uint32_t>(api::utils::div_up(depth, INT64_C(4))),
  };
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

//
// vTensor
//

vTensor::vTensor(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options)
  : view_(new vTensorStorage{
      context,
      sizes,
      options,
    }) {
}

api::VulkanImage& vTensor::image(
    api::Command::Buffer& command_buffer,
    const api::PipelineStageFlags stage) const & {
  //view_->transition(command_buffer, stage, api::MemoryAccessType::READ);

  return view_->image_;
}

api::VulkanImage& vTensor::image(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage) const & {
  view_->transition(pipeline_barrier, stage, api::MemoryAccessType::READ);

  return view_->image_;
}

api::VulkanImage& vTensor::image(
    api::Command::Buffer& command_buffer,
    const api::PipelineStageFlags stage,
    const api::MemoryAccessFlags access) & {
  //view_->transition(command_buffer, stage, access);

  return view_->image_;
}

api::VulkanImage& vTensor::image(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage,
    const api::MemoryAccessFlags access) & {
  view_->transition(pipeline_barrier, stage, access);

  return view_->image_;
}

//
// vTensorStorage
//

api::VulkanImage allocate_image(
    api::Context* const context_ptr, api::utils::uvec3& extents) {
  api::Adapter* adapter_ptr = context_ptr->adapter_ptr();

  api::ImageSampler::Properties sampler_props{
    VK_FILTER_NEAREST,
    VK_SAMPLER_MIPMAP_MODE_NEAREST,
    VK_SAMPLER_ADDRESS_MODE_REPEAT,
    VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
  };

  VkSampler sampler = adapter_ptr->sampler_cache().retrieve(sampler_props);

  return adapter_ptr->vma().create_image3d_fp(
      vk_extent(extents), sampler_props, sampler, true);
}

vTensorStorage::vTensorStorage(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options)
  : context_(context),
    extents_(image_extents(sizes)),
    options_(options),
    sizes_(sizes),
    strides_(sizes.size()),
    image_(allocate_image(context_, extents_)),
    last_access_{} {
  ops::verify(options);
}

vTensorStorage::~vTensorStorage() {
  context_->register_image_cleanup(image_);
}

void vTensorStorage::transition(
    api::Command::Buffer& command_buffer,
    const api::PipelineStageFlags cur_stage,
    const api::MemoryAccessFlags cur_access) {
  // Get last stage access
  api::PipelineStageFlags prev_stage = last_access_.stage;
  api::MemoryAccessFlags prev_access = last_access_.access;

  api::PipelineBarrier pipeline_barrier{};

  const VkImageLayout cur_layout = image_.layout();
  const VkImageLayout new_layout = vk_layout(cur_stage, cur_access);

  const bool layout_changed = cur_layout != new_layout;

  const bool read_requested = (cur_access & api::MemoryAccessType::READ) != 0;
  const bool prev_written = (prev_access & api::MemoryAccessType::WRITE) != 0;

  const bool is_RAW = read_requested && prev_written;

  if (is_RAW || layout_changed) {
    pipeline_barrier.stage.src |= vk_stage(prev_stage);
    pipeline_barrier.stage.dst |= vk_stage(cur_stage);

    const VkImageLayout cur_layout = image_.layout();
    const VkImageLayout new_layout = vk_layout(cur_stage, cur_access);

    pipeline_barrier.images.push_back(api::ImageMemoryBarrier(
        vk_access(prev_stage, prev_access),
        vk_access(cur_stage, cur_access),
        cur_layout,
        new_layout,
        image_));

    image_.set_layout(new_layout);
  }

  last_access_.stage = cur_stage;
  last_access_.access = cur_access;

  if (pipeline_barrier) {
    if (0u == pipeline_barrier.stage.src) {
      pipeline_barrier.stage.src = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }

    if (0u == pipeline_barrier.stage.dst) {
      pipeline_barrier.stage.src = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    command_buffer.barrier(pipeline_barrier);
  }
}

void vTensorStorage::transition(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags cur_stage,
    const api::MemoryAccessFlags cur_access) {
  // Get last stage access
  api::PipelineStageFlags prev_stage = last_access_.stage;
  api::MemoryAccessFlags prev_access = last_access_.access;

  const VkImageLayout cur_layout = image_.layout();
  const VkImageLayout new_layout = vk_layout(cur_stage, cur_access);

  const bool layout_changed = cur_layout != new_layout;
  const bool prev_written = (prev_access & api::MemoryAccessType::WRITE) != 0;

  if (prev_written || layout_changed) {
    VkPipelineStageFlags src_stage = vk_stage(prev_stage);
    if (0u == src_stage) {
      src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
    VkPipelineStageFlags dst_stage = vk_stage(cur_stage);
    if (0u == dst_stage) {
      dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    pipeline_barrier.stage.src |= src_stage;
    pipeline_barrier.stage.dst |= dst_stage;

    pipeline_barrier.images.push_back(api::ImageMemoryBarrier(
        vk_access(prev_stage, prev_access),
        vk_access(cur_stage, cur_access),
        cur_layout,
        new_layout,
        image_));

    image_.set_layout(new_layout);
  }

  last_access_.stage = cur_stage;
  last_access_.access = cur_access;
}

void add_buffer_barrier(
    api::PipelineBarrier& pipeline_barrier,
    const api::VulkanBuffer& buffer,
    const api::PipelineStageFlags prev_stage,
    const api::MemoryAccessFlags prev_access,
    const api::PipelineStageFlags cur_stage,
    const api::MemoryAccessFlags cur_access) {
  // Check for RAW
  const bool read_requested = (cur_access & api::MemoryAccessType::READ) != 0;
  const bool prev_written = (prev_access & api::MemoryAccessType::WRITE) != 0;

  const bool is_RAW = read_requested && prev_written;

  if (is_RAW) {
    VkPipelineStageFlags src_stage = vk_stage(prev_stage);
    if (0u == src_stage) {
      src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
    VkPipelineStageFlags dst_stage = vk_stage(cur_stage);
    if (0u == dst_stage) {
      dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    pipeline_barrier.stage.src |= src_stage;
    pipeline_barrier.stage.dst |= dst_stage;

    pipeline_barrier.buffers.push_back(api::BufferMemoryBarrier(
        vk_access(prev_stage, prev_access),
        vk_access(cur_stage, cur_access),
        buffer));
  }
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

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
