#ifndef THC_CACHING_ALLOCATOR_H
#define THC_CACHING_ALLOCATOR_H

#include <hip/hip_runtime.h>

namespace caffe2 {

struct THCCachingAllocatorImpl;

class THCCachingAllocator {
 public:
  THCCachingAllocator();
  ~THCCachingAllocator();

  hipError_t Alloc(void** refPtr, size_t nbytes, hipStream_t stream);
  hipError_t Free(void* ptr);

 private:
  THCCachingAllocatorImpl* _impl;
};

} // namespace caffe2

#endif
