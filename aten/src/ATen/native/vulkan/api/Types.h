#pragma once

// @lint-ignore-every CLANGTIDY bugprone-branch-clone

#ifdef USE_VULKAN_API

#include <cstddef>
#include <cstdint>

#include <ATen/native/vulkan/api/vk_api.h>

#include <ATen/native/vulkan/api/Exception.h>

#ifdef USE_VULKAN_FP16_INFERENCE
#define VK_FORMAT_FLOAT4 VK_FORMAT_R16G16B16A16_SFLOAT
#else
#define VK_FORMAT_FLOAT4 VK_FORMAT_R32G32B32A32_SFLOAT
#endif /* USE_VULKAN_FP16_INFERENCE */

#define VK_FORALL_SCALAR_TYPES(_)                        \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, Byte)              \
  _(int8_t, VK_FORMAT_R8G8B8A8_SINT, Char)               \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, Int)           \
  _(bool, VK_FORMAT_R8G8B8A8_SINT, Bool)                 \
  _(unsigned short, VK_FORMAT_R16G16B16A16_SFLOAT, Half) \
  _(float, VK_FORMAT_FLOAT4, Float)                      \
  _(int8_t, VK_FORMAT_R8G8B8A8_SINT, QInt8)              \
  _(uint8_t, VK_FORMAT_R8G8B8A8_UINT, QUInt8)            \
  _(int32_t, VK_FORMAT_R32G32B32A32_SINT, QInt32)

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// Scalar Types
//

enum class ScalarType : int8_t {
#define DEFINE_ENUM_VAL_(ctype, vkformat, name) name,
  VK_FORALL_SCALAR_TYPES(DEFINE_ENUM_VAL_)
#undef DEFINE_ENUM_VAL_
      Undefined,
  NumOptions
};

#define DEFINE_CONSTANT(ctype, vkformat, name) \
  constexpr ScalarType k##name = ScalarType::name;

VK_FORALL_SCALAR_TYPES(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

/*
 * Given a `ScalarType`, return the corresponding `VkFormat` that should be used
 * for image texture storage. The `ScalarType` to `VkFormat` mapping is dictated
 * by the `VK_FORALL_SCALAR_TYPE` macro in `api/Types.h`
 */
inline VkFormat to_vkformat(const ScalarType t) {
#define CASE_VK_FORMAT(ctype, vkformat, name) \
  case ScalarType::name:                      \
    return vkformat;

  switch (t) {
    VK_FORALL_SCALAR_TYPES(CASE_VK_FORMAT)
    default:
      VK_THROW("Unknown ScalarType: ", t);
  }
#undef CASE_VK_FORMAT
}

/*
 * Given a `VkFormat`, return the `ScalarType` that best represents the data
 * type of invidivual elements in an image texture of the `VkFormat`. Note that
 * this mapping is different from the `to_vkformat()` function, since different
 * `ScalarType`s may use the same `VkFormat`.
 */
inline ScalarType element_scalartype(const VkFormat vkformat) {
  switch (vkformat) {
    case VK_FORMAT_R8G8B8A8_SINT:
      return kChar;
    case VK_FORMAT_R8G8B8A8_UINT:
      return kByte;
    case VK_FORMAT_R32G32B32A32_SINT:
      return kInt;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return kFloat;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return kHalf;
    default:
      VK_THROW("No corresponding scalar type for unknown VkFormat: ", vkformat);
  }
}

/*
 * Given a ScalarType, return `sizeof(ctype)` where ctype is the C type
 * corresponding to the ScalarType. The C type to ScalarType mapping is dictated
 * by the VK_FORALL_SCALAR_TYPE macro in api/Types.h
 */
inline size_t element_size(const ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, vkformat, name) \
  case ScalarType::name:                             \
    return sizeof(ctype);

  switch (t) {
    VK_FORALL_SCALAR_TYPES(CASE_ELEMENTSIZE_CASE)
    default:
      VK_THROW("Unknown ScalarType: ", t);
  }
#undef CASE_ELEMENTSIZE_CASE
}

inline const char* to_string(const ScalarType t) {
#define CASE_TO_STRING(ctype, vkformat, name) \
  case ScalarType::name:                      \
    return #name;

  switch (t) {
    VK_FORALL_SCALAR_TYPES(CASE_TO_STRING)
    default:
      return "UNKNOWN_SCALAR_TYPE";
  }
#undef CASE_TO_STRING
}

inline std::ostream& operator<<(std::ostream& os, const ScalarType dtype) {
  return os << to_string(dtype);
}

//
// Map ScalarTypes to C++ types
//

template <ScalarType N>
struct ScalarTypeToCType;

#define SPECIALIZE_ScalarTypeToCType(ctype, vkformat, scalar_type) \
  template <>                                                      \
  struct ScalarTypeToCType<                                        \
      ::at::native::vulkan::api::ScalarType::scalar_type> {        \
    using type = ctype;                                            \
  };

VK_FORALL_SCALAR_TYPES(SPECIALIZE_ScalarTypeToCType)

#undef SPECIALIZE_ScalarTypeToCPPType

//
// GPU Storage Options
//

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
