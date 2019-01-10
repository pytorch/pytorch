#ifndef THC_CACHING_ALLOCATOR_H
#define THC_CACHING_ALLOCATOR_H

#include <cuda_runtime.h>

namespace caffe2 {

struct THCCachingAllocatorImpl;

class THCCachingAllocator {
 public:
  THCCachingAllocator();
  ~THCCachingAllocator();

  cudaError_t Alloc(void** refPtr, size_t nbytes, cudaStream_t stream);
  cudaError_t Free(void* ptr);

 private:
  THCCachingAllocatorImpl* _impl;
};

} // namespace caffe2

#endif
