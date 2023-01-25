#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Tensor.h>

namespace at {
namespace native {
namespace vulkan {

/*
 * Input and output tensors require staging buffers when transferring data to
 * and from the CPU. This struct is a straightforward pair of a vTensor and a
 * StorageBuffer to provide an easy interface for moving data into and out of
 * input/output tensors.
 */
struct TensorStaging {
 public:
  explicit TensorStaging(
      api::Context* context,
      IntArrayRef sizes,
      c10::ScalarType dtype,
      api::StorageType storage_type = api::StorageType::TEXTURE_3D,
      c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

  TensorStaging(TensorStaging&&) noexcept;

  ~TensorStaging() {}

  vTensor tensor;
  // Keep the StorageBuffer in a unique_ptr to enable move construction so that
  // TensorStaging can be stored in container types.
  std::unique_ptr<api::StorageBuffer> staging;

  //
  // Data pointer to/from staging
  //

  void ptr_to_staging(void* src);
  void staging_to_ptr(void* dst);

  //
  // Staging to/from GPU
  //

  void record_copy_to_gpu(api::Context* context);
  void record_copy_from_gpu(api::Context* context);

 private:
  void memcpy_to_mapping(void* src, api::MemoryMap& dst_mapping);
  void memcpy_from_mapping(api::MemoryMap& src_mapping, void* dst);
};

//
// Utility functions for memcpy
//

template <typename T>
void memcpy_to_mapping_impl(
    void* src,
    api::MemoryMap& dst_mapping,
    size_t nbytes) {
  T* data_ptr = dst_mapping.template data<T>();
  memcpy(data_ptr, reinterpret_cast<T*>(src), nbytes);
}

template <typename T>
void memcpy_from_mapping_impl(
    api::MemoryMap& src_mapping,
    void* dst,
    size_t nbytes) {
  T* data_ptr = src_mapping.template data<T>();
  memcpy(reinterpret_cast<T*>(dst), data_ptr, nbytes);
}

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
