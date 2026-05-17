#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/api/Utils.h>

namespace at {
namespace native {
namespace vulkan {

namespace {

/*
 * Calculates the strides of a contiguous tensor. empty_tensor_restride from
 * TensorImpl.h was used as a reference.
 */
std::vector<int64_t> calc_contiguous_strides(
    const std::vector<int64_t>& sizes) {
  int64_t ndim = static_cast<int64_t>(sizes.size());
  std::vector<int64_t> strides(ndim);

  int64_t running_product = 1;
  if (ndim >= 1) {
    strides.at(ndim - 1) = running_product;
    for (int i = static_cast<int>(sizes.size()) - 2; i >= 0; --i) {
      running_product *= sizes.at(i + 1);
      strides.at(i) = running_product;
    }
  }

  return strides;
}

std::vector<int64_t> calc_channels_last_strides(
    const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size());

  switch (sizes.size()) {
    case 4:
      strides.at(1) = 1;
      strides.at(3) = sizes.at(1);
      strides.at(2) = strides.at(3) * sizes.at(3);
      strides.at(0) = strides.at(2) * sizes.at(2);
      return strides;
    case 3:
      strides.at(0) = 1;
      strides.at(2) = sizes.at(0);
      strides.at(1) = strides.at(2) * sizes.at(2);
      return strides;
    default:
      VK_THROW("ChannelsLast format only available for 3 <= ndim <= 4!");
  }

  return strides;
}

/*
 * Calculates the strides of a tensor based on the sizes and memory format. Note
 * that strides are only valid for vTensors that are backed by buffer storage;
 * if texture storage is used then the strides are invalid and set to zeros.
 */
std::vector<int64_t> calc_strides(
    const std::vector<int64_t>& sizes,
    const api::GPUMemoryLayout memory_layout,
    const api::StorageType storage_type) {
  switch (storage_type) {
    case api::StorageType::BUFFER:
      switch (memory_layout) {
        case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
          return calc_contiguous_strides(sizes);
          break;
        case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
          return calc_channels_last_strides(sizes);
          break;
        default:
          VK_THROW("Invalid memory format used to create vTensor!");
      }
      break;
    case api::StorageType::TEXTURE_3D:
    case api::StorageType::TEXTURE_2D:
      return std::vector<int64_t>(sizes.size());
    default:
      VK_THROW("Invalid storage type used to create vTensor!");
  }
}

/*
 * When stored on the GPU, one dimension will be aligned to the next multiple of
 * 4 in order to take advantage of vec4 data types. The dimension that is
 * packed is denoted by the GPUMemoryLayout. This function adjusts one of
 * the dimensions based on the desired memory format and storage type and
 * returns a sizes array describing the dimensions of the memory used to store
 * the tensor data on the GPU.
 */
std::vector<int64_t> calc_gpu_sizes(
    const std::vector<int64_t>& sizes,
    const api::GPUMemoryLayout memory_layout,
    const api::StorageType storage_type) {
  VK_CHECK_COND(storage_type != api::StorageType::UNKNOWN);

  std::vector<int64_t> gpu_sizes;
  if (storage_type == api::StorageType::BUFFER) {
    gpu_sizes.resize(sizes.size());
    for (size_t i = 0; i < sizes.size(); i++) {
      gpu_sizes.at(i) = sizes.at(i);
    }
  }
  // For texture storage, tensors are typically stored using 3D image textures.
  // Batches are stacked along the depth dimension. To represent the physical
  // 3 dimensionality of the image texture (with concatenated batches) GPU sizes
  // will be fixed to 4 dimensions when using texture storage.
  else {
    VK_CHECK_COND(
        sizes.size() >= 0 && sizes.size() <= 4,
        "Texture storage only valid for 0 <= ndim <= 4, received: ",
        sizes.size());

    gpu_sizes.resize(4);
    gpu_sizes.at(0) = api::utils::val_at(-4, sizes);
    gpu_sizes.at(1) = api::utils::val_at(-3, sizes);
    gpu_sizes.at(2) = api::utils::val_at(-2, sizes);
    gpu_sizes.at(3) = api::utils::val_at(-1, sizes);
  }

  size_t ndim = gpu_sizes.size();
  switch (memory_layout) {
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      if (ndim >= 1) {
        gpu_sizes.at(ndim - 1) =
            api::utils::align_up(api::utils::val_at(-1, sizes), INT64_C(4));
      }
      break;

    case api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED:
      if (ndim >= 2) {
        gpu_sizes.at(ndim - 2) =
            api::utils::align_up(api::utils::val_at(-2, sizes), INT64_C(4));
      }
      break;

    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      if (ndim >= 3) {
        gpu_sizes.at(ndim - 3) =
            api::utils::align_up(api::utils::val_at(-3, sizes), INT64_C(4));
      }
      break;
  }

  return gpu_sizes;
}

/*
 * Creates a uvec3 denoting the extents of the image texture that will be
 * created to store a tensor of a given size.
 */
api::utils::uvec3 create_image_extents(
    const std::vector<int64_t>& gpu_sizes,
    const api::StorageType storage_type,
    const api::GPUMemoryLayout memory_layout) {
  size_t ndim = gpu_sizes.size();

  if (storage_type == api::StorageType::BUFFER) {
    // image extents do not apply to buffer storage
    return {0u, 0u, 0u};
  } else {
    VK_CHECK_COND(
        ndim >= 1 && ndim <= 4,
        "Texture storage only valid for 1 <= ndim <= 4!");

    using namespace api::utils;
    uint32_t width = safe_downcast<uint32_t>(val_at(-1, gpu_sizes));
    uint32_t height = safe_downcast<uint32_t>(val_at(-2, gpu_sizes));
    uint32_t channels = safe_downcast<uint32_t>(val_at(-3, gpu_sizes));
    uint32_t batch = safe_downcast<uint32_t>(val_at(-4, gpu_sizes));

    switch (memory_layout) {
      case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
        VK_CHECK_COND(width % 4 == 0, "Channels must be divisible by 4!");
        width /= 4;
        break;
      case api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED:
        VK_CHECK_COND(height % 4 == 0, "Channels must be divisible by 4!");
        height /= 4;
        break;
      case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
        VK_CHECK_COND(channels % 4 == 0, "Channels must be divisible by 4!");
        channels /= 4;
        break;
      default:
        VK_THROW("Invalid memory format used!");
    }

    return {width, height, batch * channels};
  }
}

api::UniformParamsBuffer make_metadata_uniform(
    api::Context* const context,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    const api::StorageType storage_type) {
  if (storage_type != api::StorageType::BUFFER) {
    return api::UniformParamsBuffer();
  }

  vTensor::BufferMetadata metadata{
      api::utils::make_whcn_uvec4(sizes),
      api::utils::make_whcn_uvec4(strides),
      api::utils::safe_downcast<uint32_t>(sizes.size()),
      api::utils::safe_downcast<uint32_t>(api::utils::multiply_integers(sizes)),
  };

  return api::UniformParamsBuffer(context, metadata);
}

} // namespace

//
// vTensor
//

vTensor::vTensor(
    api::Context* const context,
    const std::vector<int64_t>& sizes,
    const api::ScalarType dtype,
    const api::StorageType storage_type,
    const api::GPUMemoryLayout memory_layout,
    const bool allocate_memory)
    : dtype_(dtype),
      memory_layout_(memory_layout),
      // Calculate sizes and strides
      sizes_(sizes.begin(), sizes.end()),
      strides_{calc_strides(sizes, memory_layout_, storage_type)},
      gpu_sizes_{calc_gpu_sizes(sizes, memory_layout_, storage_type)},
      gpu_strides_{calc_strides(gpu_sizes_, memory_layout_, storage_type)},
      virtual_extents_(
          create_image_extents(gpu_sizes_, storage_type, memory_layout)),
      // Utility Uniform Buffers that can be passed to shaders as arguments
      metadata_uniform_(),
      cpu_sizes_uniform_(nullptr),
      gpu_sizes_uniform_(nullptr),
      extents_uniform_(nullptr),
      // Construct Tensor storage
      view_(std::make_shared<vTensorStorage>(
          context,
          storage_type,
          memory_layout_,
          gpu_sizes_,
          dtype_,
          allocate_memory)) {}

vTensor::vTensor(
    api::Context* const context,
    const std::vector<int64_t>& sizes,
    double q_scale,
    int64_t q_zero_point,
    const api::ScalarType dtype,
    const api::StorageType storage_type,
    const api::GPUMemoryLayout memory_layout)
    : dtype_(dtype),
      memory_layout_(memory_layout),
      // Calculate sizes and strides
      sizes_(sizes.begin(), sizes.end()),
      strides_{calc_strides(sizes, memory_layout_, storage_type)},
      gpu_sizes_{calc_gpu_sizes(sizes, memory_layout_, storage_type)},
      gpu_strides_{calc_strides(gpu_sizes_, memory_layout_, storage_type)},
      virtual_extents_(
          create_image_extents(gpu_sizes_, storage_type, memory_layout)),
      // Vulkan uniform buffer containing sizes and stride info
      metadata_uniform_(),
      cpu_sizes_uniform_(nullptr),
      gpu_sizes_uniform_(nullptr),
      extents_uniform_(nullptr),
      // Quantization params
      is_quantized_{true},
      q_scale_{q_scale},
      q_zero_point_{q_zero_point},
      // Construct Tensor storage
      view_(std::make_shared<vTensorStorage>(
          context,
          storage_type,
          memory_layout_,
          gpu_sizes_,
          dtype_)) {}

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

api::VulkanBuffer& vTensor::buffer(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage) const& {
  view_->transition(pipeline_barrier, stage, api::MemoryAccessType::READ);
  return view_->buffer_;
}

api::VulkanBuffer& vTensor::buffer(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage,
    const api::MemoryAccessFlags access) & {
  view_->transition(pipeline_barrier, stage, access);
  return view_->buffer_;
}

api::VulkanBuffer& vTensor::buffer_metadata() {
  if (!metadata_uniform_.buffer()) {
    metadata_uniform_ = make_metadata_uniform(
        view_->context_, gpu_sizes_, gpu_strides_, storage_type());
  }
  return metadata_uniform_.buffer();
}

std::shared_ptr<api::UniformParamsBuffer> vTensor::cpu_sizes_ubo() {
  if (!cpu_sizes_uniform_) {
    cpu_sizes_uniform_.reset(new api::UniformParamsBuffer(
        view_->context_, api::utils::make_whcn_ivec4(sizes_)));
  }
  return cpu_sizes_uniform_;
}

std::shared_ptr<api::UniformParamsBuffer> vTensor::gpu_sizes_ubo() {
  if (!gpu_sizes_uniform_) {
    gpu_sizes_uniform_.reset(new api::UniformParamsBuffer(
        view_->context_, api::utils::make_whcn_ivec4(gpu_sizes_)));
  }
  return gpu_sizes_uniform_;
}

std::shared_ptr<api::UniformParamsBuffer> vTensor::extents_ubo() {
  if (!extents_uniform_) {
    extents_uniform_.reset(new api::UniformParamsBuffer(
        view_->context_,
        api::utils::uvec4(
            {view_->extents_.data[0],
             view_->extents_.data[1],
             view_->extents_.data[2],
             1u})));
  }
  return extents_uniform_;
}

vTensor::BufferMetadata vTensor::get_cpu_buffer_metadata() const {
  return {
      api::utils::make_whcn_uvec4(sizes_),
      api::utils::make_whcn_uvec4(strides_),
      api::utils::safe_downcast<uint32_t>(sizes_.size()),
      api::utils::safe_downcast<uint32_t>(
          api::utils::multiply_integers(sizes_)),
  };
}

VmaAllocationCreateInfo vTensor::get_allocation_create_info() const {
  switch (storage_type()) {
    case api::StorageType::BUFFER:
      return view_->buffer_.allocation_create_info();
    case api::StorageType::TEXTURE_2D:
    case api::StorageType::TEXTURE_3D:
      return view_->image_.allocation_create_info();
    case api::StorageType::UNKNOWN:
      break;
  }
  return {};
}

VkMemoryRequirements vTensor::get_memory_requirements() const {
  switch (storage_type()) {
    case api::StorageType::BUFFER:
      return view_->buffer_.get_memory_requirements();
    case api::StorageType::TEXTURE_2D:
    case api::StorageType::TEXTURE_3D:
      return view_->image_.get_memory_requirements();
    case api::StorageType::UNKNOWN:
      break;
  }
  return {};
}

void vTensor::bind_allocation(const api::MemoryAllocation& allocation) {
  switch (storage_type()) {
    case api::StorageType::BUFFER:
      view_->buffer_.bind_allocation(allocation);
      break;
    case api::StorageType::TEXTURE_2D:
    case api::StorageType::TEXTURE_3D:
      view_->image_.bind_allocation(allocation);
      break;
    case api::StorageType::UNKNOWN:
      break;
  }
}

void vTensor::update_size_metadata(const std::vector<int64_t>& new_sizes) {
  sizes_ = new_sizes;
  gpu_sizes_ = calc_gpu_sizes(sizes_, memory_layout_, storage_type());
  virtual_extents_ =
      create_image_extents(gpu_sizes_, storage_type(), memory_layout_);

  if (cpu_sizes_uniform_) {
    cpu_sizes_uniform_->update(api::utils::make_whcn_ivec4(sizes_));
  }

  if (gpu_sizes_uniform_) {
    gpu_sizes_uniform_->update(api::utils::make_whcn_ivec4(gpu_sizes_));
  }

  if (extents_uniform_) {
    extents_uniform_->update(api::utils::uvec4(
        {virtual_extents_.data[0],
         virtual_extents_.data[1],
         virtual_extents_.data[2],
         1u}));
  }
}

void vTensor::reallocate(const std::vector<int64_t>& new_sizes) {
  update_size_metadata(new_sizes);
  view_->discard_and_reallocate(
      calc_gpu_sizes(new_sizes, memory_layout_, storage_type()),
      memory_layout_,
      dtype_);
}

void vTensor::virtual_resize(const std::vector<int64_t>& new_sizes) {
  update_size_metadata(new_sizes);
  if (storage_type() == api::StorageType::BUFFER) {
    if (gpu_nbytes() > view_->buffer_.mem_size()) {
      VK_THROW(
          "Cannot virtual_resize a vTensor with sizes that require a larger "
          "buffer! reallocate() should be used instead.");
    }
  } else {
    bool valid_resize = true;
    if (virtual_extents_.data[0] > view_->extents_.data[0]) {
      valid_resize = false;
    }
    if (virtual_extents_.data[1] > view_->extents_.data[1]) {
      valid_resize = false;
    }
    if (virtual_extents_.data[2] > view_->extents_.data[2]) {
      valid_resize = false;
    }

    if (!valid_resize) {
      VK_THROW(
          "Cannot virtual_resize a vTensor with sizes that require a larger "
          "image texture! reallocate() should be used instead.");
    }
  }
}

//
// vTensorStorage
//

static api::VulkanImage allocate_image(
    api::Context* const context_ptr,
    api::utils::uvec3& extents,
    const api::StorageType storage_type,
    const VkFormat image_format,
    const bool allocate_memory) {
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
    default:
      // Return an empty VulkanImage by default
      return api::VulkanImage();
  }

  VkSampler sampler = adapter_ptr->sampler_cache().retrieve(sampler_props);

  return adapter_ptr->vma().create_image(
      api::create_extent3d(extents),
      image_format,
      image_type,
      image_view_type,
      sampler_props,
      sampler,
      /*allow_transfer = */ true,
      /*allocate_memory = */ allocate_memory);
}

static api::VulkanBuffer allocate_buffer(
    api::Context* const context_ptr,
    const int64_t numel,
    const api::StorageType storage_type,
    const api::ScalarType dtype,
    const bool allocate_memory) {
  api::Adapter* adapter_ptr = context_ptr->adapter_ptr();

  switch (storage_type) {
    case api::StorageType::BUFFER:
      break;
    default:
      // Return an empty VulkanBuffer if Buffer storage is not used
      return api::VulkanBuffer();
  }

  return adapter_ptr->vma().create_storage_buffer(
      api::element_size(dtype) * numel, /*gpu_only = */ true, allocate_memory);
}

vTensorStorage::vTensorStorage(
    api::Context* const context,
    const api::StorageType storage_type,
    const api::GPUMemoryLayout gpu_memory_layout,
    const std::vector<int64_t>& gpu_sizes,
    const api::ScalarType dtype,
    const bool allocate_memory)
    : context_(context),
      storage_type_{storage_type},
      extents_(
          create_image_extents(gpu_sizes, storage_type, gpu_memory_layout)),
      buffer_length_{api::utils::multiply_integers(gpu_sizes)},
      image_(allocate_image(
          context_,
          extents_,
          storage_type_,
          api::to_vkformat(dtype),
          allocate_memory)),
      buffer_(allocate_buffer(
          context_,
          buffer_length_,
          storage_type_,
          dtype,
          allocate_memory)),
      last_access_{} {}

vTensorStorage::~vTensorStorage() {
  flush();
}

void vTensorStorage::flush() {
  if (image_) {
    context_->register_image_cleanup(image_);
  } else if (buffer_) {
    context_->register_buffer_cleanup(buffer_);
  }
  last_access_ = {};
}

void vTensorStorage::transition(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags cur_stage,
    const api::MemoryAccessFlags cur_access) {
  // Get last stage access
  api::PipelineStageFlags prev_stage = last_access_.stage;
  api::MemoryAccessFlags prev_access = last_access_.access;

  const bool prev_written = (prev_access & api::MemoryAccessType::WRITE) != 0;

  VkImageLayout cur_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  bool layout_changed = false;
  if (image_) {
    cur_layout = image_.layout();
    new_layout = api::vk_layout(cur_stage, cur_access);

    layout_changed = cur_layout != new_layout;
  }

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

    if (image_) {
      pipeline_barrier.images.emplace_back(
          api::vk_access(prev_stage, prev_access),
          api::vk_access(cur_stage, cur_access),
          cur_layout,
          new_layout,
          image_);

      image_.set_layout(new_layout);
    } else if (buffer_) {
      pipeline_barrier.buffers.emplace_back(
          api::vk_access(prev_stage, prev_access),
          api::vk_access(cur_stage, cur_access),
          buffer_);
    }
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

    pipeline_barrier.buffers.emplace_back(
        api::vk_access(prev_stage, prev_access),
        api::vk_access(cur_stage, cur_access),
        buffer);
  }
}

void vTensorStorage::discard_and_reallocate(
    const std::vector<int64_t>& gpu_sizes,
    const api::GPUMemoryLayout gpu_memory_layout,
    const api::ScalarType dtype) {
  const bool image_owns_memory = image_.owns_memory();
  const bool buffer_owns_memory = buffer_.owns_memory();

  flush();

  extents_ = create_image_extents(gpu_sizes, storage_type_, gpu_memory_layout);
  image_ = allocate_image(
      context_,
      extents_,
      storage_type_,
      api::to_vkformat(dtype),
      image_owns_memory);

  buffer_length_ = api::utils::multiply_integers(gpu_sizes);
  buffer_ = allocate_buffer(
      context_, buffer_length_, storage_type_, dtype, buffer_owns_memory);
}

} // namespace vulkan
} // namespace native
} // namespace at
