#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Tensor.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

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
      api::utils::safe_downcast<uint32_t>(
          api::utils::div_up(depth, INT64_C(4))),
  };
}

} // namespace

//
// vTensor
//

vTensor::vTensor(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options)
    : view_(std::make_shared<vTensorStorage>(
          context,
          sizes,
          api::StorageType::TEXTURE_3D,
          options)) {}

vTensor::vTensor(
    api::Context* const context,
    const IntArrayRef sizes,
    const api::StorageType storage_type,
    const TensorOptions& options)
    : view_(std::make_shared<vTensorStorage>(
          context,
          sizes,
          storage_type,
          options)) {}

vTensor::vTensor(
    api::Context* const context,
    const IntArrayRef sizes,
    const TensorOptions& options,
    double q_scale,
    int64_t q_zero_point)
    : view_(std::make_shared<vTensorStorage>(
          context,
          sizes,
          api::StorageType::TEXTURE_3D,
          options,
          q_scale,
          q_zero_point)) {}

vTensor::vTensor(
    api::Context* const context,
    const IntArrayRef sizes,
    const api::StorageType storage_type,
    const TensorOptions& options,
    double q_scale,
    int64_t q_zero_point)
    : view_(std::make_shared<vTensorStorage>(
          context,
          sizes,
          storage_type,
          options,
          q_scale,
          q_zero_point)) {}

api::VulkanImage& vTensor::image(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage) const& {
  view_->transition(pipeline_barrier, stage, api::MemoryAccessType::READ);

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
    api::Context* const context_ptr,
    api::utils::uvec3& extents,
    api::StorageType storage_type,
    const VkFormat image_format) {
  api::Adapter* adapter_ptr = context_ptr->adapter_ptr();

  api::ImageSampler::Properties sampler_props{
      VK_FILTER_NEAREST,
      VK_SAMPLER_MIPMAP_MODE_NEAREST,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
  };

  VkImageType image_type = VK_IMAGE_TYPE_3D;
  VkImageViewType image_view_type = VK_IMAGE_VIEW_TYPE_3D;

  switch (storage_type) {
    case api::StorageType::TEXTURE_3D:
      image_type = VK_IMAGE_TYPE_3D;
      image_view_type = VK_IMAGE_VIEW_TYPE_3D;
      break;
    case api::StorageType::TEXTURE_2D:
      image_type = VK_IMAGE_TYPE_2D;
      image_view_type = VK_IMAGE_VIEW_TYPE_2D;
      break;
    case api::StorageType::BUFFER:
    case api::StorageType::UNKNOWN:
      TORCH_CHECK(false, "Requested storage type must be a texture type.");
  }

  VkSampler sampler = adapter_ptr->sampler_cache().retrieve(sampler_props);

  return adapter_ptr->vma().create_image(
      api::create_extent3d(extents),
      image_format,
      image_type,
      image_view_type,
      sampler_props,
      sampler,
      true);
}

vTensorStorage::vTensorStorage(
    api::Context* const context,
    const IntArrayRef sizes,
    const api::StorageType storage_type,
    const TensorOptions& options)
    : context_(context),
      extents_(image_extents(sizes)),
      options_(options),
      sizes_(sizes),
      strides_(sizes.size()),
      storage_type_{storage_type},
      image_(allocate_image(
          context_,
          extents_,
          storage_type_,
          api::vk_format(options_.dtype()))),
      last_access_{} {
  ops::verify(options);
}

vTensorStorage::vTensorStorage(
    api::Context* const context,
    const IntArrayRef sizes,
    const api::StorageType storage_type,
    const TensorOptions& options,
    double q_scale_in,
    int64_t q_zero_point_in)
    : context_(context),
      extents_(image_extents(sizes)),
      options_(options),
      sizes_(sizes),
      strides_(sizes.size()),
      is_quantized_{true},
      q_scale{q_scale_in},
      q_zero_point{q_zero_point_in},
      storage_type_{storage_type},
      image_(allocate_image(
          context_,
          extents_,
          storage_type_,
          api::vk_format(options_.dtype()))),
      last_access_{} {
  ops::verify(options);
}

vTensorStorage::~vTensorStorage() {
  context_->register_image_cleanup(image_);
}

void vTensorStorage::transition(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags cur_stage,
    const api::MemoryAccessFlags cur_access) {
  // Get last stage access
  api::PipelineStageFlags prev_stage = last_access_.stage;
  api::MemoryAccessFlags prev_access = last_access_.access;

  const VkImageLayout cur_layout = image_.layout();
  const VkImageLayout new_layout = api::vk_layout(cur_stage, cur_access);

  const bool layout_changed = cur_layout != new_layout;
  const bool prev_written = (prev_access & api::MemoryAccessType::WRITE) != 0;

  if (prev_written || layout_changed) {
    VkPipelineStageFlags src_stage = api::vk_stage(prev_stage);
    if (0u == src_stage) {
      src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
    VkPipelineStageFlags dst_stage = api::vk_stage(cur_stage);
    if (0u == dst_stage) {
      dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    pipeline_barrier.stage.src |= src_stage;
    pipeline_barrier.stage.dst |= dst_stage;

    pipeline_barrier.images.push_back(api::ImageMemoryBarrier(
        api::vk_access(prev_stage, prev_access),
        api::vk_access(cur_stage, cur_access),
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
    VkPipelineStageFlags src_stage = api::vk_stage(prev_stage);
    if (0u == src_stage) {
      src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
    VkPipelineStageFlags dst_stage = api::vk_stage(cur_stage);
    if (0u == dst_stage) {
      dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    pipeline_barrier.stage.src |= src_stage;
    pipeline_barrier.stage.dst |= dst_stage;

    pipeline_barrier.buffers.push_back(api::BufferMemoryBarrier(
        api::vk_access(prev_stage, prev_access),
        api::vk_access(cur_stage, cur_access),
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
