#pragma once

#ifdef USE_VULKAN_API

#include <string.h>

#include <ATen/native/vulkan/graph/Graph.h>

namespace at {
namespace native {
namespace vulkan {

//
// Functions to memcpy data into staging buffer
//

void memcpy_to_mapping(
    const void* src,
    api::MemoryMap& dst_mapping,
    const size_t nbytes,
    const api::ScalarType dtype);
void memcpy_from_mapping(
    const api::MemoryMap& src_mapping,
    void* dst,
    const size_t nbytes,
    const api::ScalarType dtype);

//
// Utility functions for memcpy
//

template <typename T>
void memcpy_to_mapping_impl(
    const void* src,
    api::MemoryMap& dst_mapping,
    const size_t nbytes) {
  T* data_ptr = dst_mapping.template data<T>();
  memcpy(data_ptr, reinterpret_cast<const T*>(src), nbytes);
}

template <typename T>
void memcpy_from_mapping_impl(
    api::MemoryMap& src_mapping,
    void* dst,
    const size_t nbytes) {
  T* data_ptr = src_mapping.template data<T>();
  memcpy(reinterpret_cast<T*>(dst), data_ptr, nbytes);
}

//
// Functions to copy data into and out of a staging buffer
//

void copy_ptr_to_staging(
    const void* src,
    api::StorageBuffer& staging,
    const size_t nbytes);
void copy_staging_to_ptr(
    api::StorageBuffer& staging,
    void* dst,
    const size_t nbytes);

//
// Functions to record copying data between a staging buffer and a vTensor
//

void encode_copy_to_vtensor(
    api::Context* context,
    api::StorageBuffer& staging,
    vTensor& tensor);
void encode_copy_from_vtensor(
    api::Context* context,
    vTensor& tensor,
    api::StorageBuffer& staging);

/*
 * OpNode that allows copying data into and out of a staging buffer.
 */
class StagingNode : public virtual OpNode {
 public:
  explicit StagingNode(ValueRef from, ValueRef to);

  void encode_execute(ComputeGraph* graph) const override;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
