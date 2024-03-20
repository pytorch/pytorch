#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

void transfer_cpu_to_vulkan(const Tensor&, vTensor&);

void transfer_vulkan_to_cpu(vTensor&, Tensor&);

void pack_cpu_to_vulkan(const Tensor& src, vTensor& dst);

void pack_vulkan_to_cpu(vTensor& src, Tensor& dst);

Tensor& copy_(Tensor& dst, const Tensor& src);

vTensor to_vulkan(
    at::Tensor& src,
    const api::StorageType storage_type = api::StorageType::TEXTURE_3D);

at::Tensor from_vulkan(vTensor& v_src);

//
// Utility functions for memcpy
//

template <typename T>
void memcpy_to_mapping_impl(const Tensor& src, api::MemoryMap& dst_mapping) {
  T* data_ptr = dst_mapping.template data<T>();
  memcpy(
      data_ptr,
      src.const_data_ptr<T>(),
      std::min(src.nbytes(), dst_mapping.nbytes()));
}

template <typename T>
void memcpy_from_mapping_impl(api::MemoryMap& src_mapping, Tensor& dst) {
  T* data_ptr = src_mapping.template data<T>();
  memcpy(
      dst.mutable_data_ptr<T>(),
      data_ptr,
      std::min(src_mapping.nbytes(), dst.nbytes()));
}

inline void memcpy_from_mapping_bool(api::MemoryMap& src_mapping, Tensor& dst) {
  uint8_t* src_ptr = src_mapping.template data<uint8_t>();
  bool* dst_ptr = dst.mutable_data_ptr<bool>();
  for (int i = 0; (unsigned)i < std::min(src_mapping.nbytes(), dst.nbytes());
       ++i) {
    dst_ptr[i] = static_cast<bool>(src_ptr[i]);
  }
}

inline void memcpy_to_mapping_uint8(
    const Tensor& src,
    api::MemoryMap& dst_mapping) {
  bool* src_ptr = src.mutable_data_ptr<bool>();
  uint8_t* dst_ptr = dst_mapping.template data<uint8_t>();
  for (int i = 0; (unsigned)i < std::min(dst_mapping.nbytes(), src.nbytes());
       ++i) {
    dst_ptr[i] = static_cast<uint8_t>(src_ptr[i]);
  }
}

void memcpy_to_mapping(const Tensor& src, api::MemoryMap& dst_mapping);

void memcpy_from_mapping(api::MemoryMap& src_mapping, Tensor& dst);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
