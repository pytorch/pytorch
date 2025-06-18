#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>

namespace c10::cuda::CUDACachingAllocator {

enum class Expandable_Segments_Handle_Type : int {
  UNSPECIFIED = 0,
  POSIX_FD = 1,
  FABRIC_HANDLE = 2,
};

// Keep this for backwards compatibility
class C10_CUDA_API CUDAAllocatorConfig
    : public CachingAllocator::AllocatorConfig {
 public:
  static bool expandable_segments() {
    return CachingAllocator::AllocatorConfig::use_expandable_segments();
  }

  static Expandable_Segments_Handle_Type expandable_segments_handle_type() {
    return instance().m_expandable_segments_handle_type;
  }

  static void set_expandable_segments_handle_type(
      Expandable_Segments_Handle_Type handle_type) {
    instance().m_expandable_segments_handle_type = handle_type;
  }

  static bool release_lock_on_cudamalloc() {
    return CachingAllocator::AllocatorConfig::
        use_release_lock_on_device_malloc();
  }

  /** Pinned memory allocator settings */
  static bool pinned_use_cuda_host_register() {
    return CachingAllocator::AllocatorConfig::pinned_use_device_host_register();
  }

 private:
  std::atomic<Expandable_Segments_Handle_Type>
      m_expandable_segments_handle_type{
          Expandable_Segments_Handle_Type::UNSPECIFIED};
};

// Keep this for backwards compatibility
using c10::CachingAllocator::setAllocatorSettings;

} // namespace c10::cuda::CUDACachingAllocator
