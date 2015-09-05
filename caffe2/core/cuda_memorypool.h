#ifndef CAFFE2_CORE_CUDA_MEMORYPOOL_H_
#define CAFFE2_CORE_CUDA_MEMORYPOOL_H_

#include <cstddef>

#include "caffe2/core/common_gpu.h"
#include "glog/logging.h"

namespace caffe2 {

class CudaMemoryPool {
 public:
  // Initializes the memory pool on the device ids, and pre-preserves the given
  // proportion of the currently free memory on the device.
  static bool InitializeMemoryPool(
      const vector<int>& device_ids,
      const float proportion_of_memory_to_reserve);

  // Finalizes the memory pool. This has to be called after all memory allocated
  // by the memory pool has been freed.
  static bool FinalizeMemoryPool();

  static inline bool MemoryPoolInitialized() { return is_memory_pool_setup_; }
  static inline bool MemoryPoolAvailableForDevice(int device_id) {
    return (device_id < memory_pool_available_for_device_.size() &&
            memory_pool_available_for_device_[device_id]);
  }

  static inline void* New(size_t nbytes) {
    if (is_memory_pool_setup_) {
      return NewWithMemoryPool(nbytes);
    } else {
      // If memory pool is not set up, use simple cudaMalloc.
      void* dev_ptr;
      CUDA_CHECK(cudaMalloc(&dev_ptr, nbytes));
      memory_allocated_before_setup_ = true;
      return dev_ptr;
    }
  }

  static inline void Delete(void* data) {
    if (is_memory_pool_setup_) {
      DeleteWithMemoryPool(data);
    } else {
      // If memory pool is not set up, use simple cudaFree.
      cudaError_t error = cudaFree(data);
      // For some reason, in Python runtime we sometimes delete a data pointer
      // after the cuda runtime exits - this is odd but is probably caused by
      // a static workspace that pycaffe2 uses, and the destruction got
      // entangled in some race condition. Anyway, since cuda runtime is exiting
      // anyway, we will not need to worry about memory leak, so we basically
      // ignore it. This is definitely not ideal but works for now.
      if (error != cudaSuccess && error != cudaErrorCudartUnloading) {
        LOG(FATAL) << "Error at: " << __FILE__ << ":" << __LINE__ << ": "
                   << cudaGetErrorString(error);
      }
    }
  }

 private:
  // CudaMemoryPool is a singleton, so it should not be instantiated.
  CudaMemoryPool() {}
  static void* NewWithMemoryPool(size_t nbytes);
  static void DeleteWithMemoryPool(void* data);

  static bool is_memory_pool_setup_;
  static bool memory_allocated_before_setup_;
  static vector<bool> memory_pool_available_for_device_;
  static vector<cudaStream_t> per_device_streams_;
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_CUDA_MEMORYPOOL_H_
