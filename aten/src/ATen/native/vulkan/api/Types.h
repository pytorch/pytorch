#pragma once

#ifdef USE_VULKAN_API
namespace at {
namespace native {
namespace vulkan {
namespace api {

/**
 * The enum below is used to describe what type of GPU memory will be used to
 * store a particular tensor's data.
 *
 * BUFFER means that a SSBO (Shader Storage Buffer Object) will be used.
 * TEXTURE_3D means that a 3-dimensional image texture will be used.
 * TEXTURE_2D means that a 2-dimensional image texture will be used.
 *
 * UNKNOWN is not expected to be used.
 */
enum class StorageType {
  BUFFER,
  TEXTURE_3D,
  TEXTURE_2D,
  UNKNOWN,
};

/**
 * The enum below is used to describe how tensor data is laid out when stored in
 * GPU memory. The name of the enum describes which dimension is tightly packed;
 * so for tensors that are stored as image textures, loading a texel will
 * retrieve 4 consecutive elements of the named dimension, and for tensors
 * stored as buffers, the named dimension will have a stride of 1.
 *
 * The GPU memory layout qualifier will be used by compute shaders to determine
 * how to convert between logical tensor coordinates and physical texel
 * coordinates. For tensors that are stored as buffers, it is expected that the
 * strides of the tensor will be used instead to convert between logical tensor
 * coordinates and linear access indices.
 */
enum class GPUMemoryLayout : uint32_t {
  TENSOR_WIDTH_PACKED = 0u,
  TENSOR_HEIGHT_PACKED = 1u,
  TENSOR_CHANNELS_PACKED = 2u,
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
