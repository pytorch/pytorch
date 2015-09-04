#include "third_party/cnmem/cnmem.h"
#include "caffe2/core/cuda_memorypool.h"

namespace caffe2 {

#define CNMEM_CHECK(condition) \
  do { \
    cnmemStatus_t error = condition; \
    CHECK_EQ(error, CNMEM_STATUS_SUCCESS) << cnmemGetErrorString(error); \
  } while (0)

bool CudaMemoryPool::is_memory_pool_setup_ = false;
bool CudaMemoryPool::memory_allocated_before_setup_ = false;
vector<bool> CudaMemoryPool::memory_pool_available_for_device_(0);
vector<cudaStream_t> CudaMemoryPool::per_device_streams_(0);

bool CudaMemoryPool::InitializeMemoryPool(
    const vector<int>& device_ids,
    const float proportion_of_memory_to_reserve) {
  if (memory_allocated_before_setup_) {
    LOG(ERROR) << "There is cuda memory allocated before we initialize the "
                  "memory pool. This should not happen: you should either "
                  "use raw cudaMalloc and cudaFree and not initialize the "
                  "pool at all, or initialize the pool before you allocate "
                  "anything.";
    return false;
  }
  if (is_memory_pool_setup_) {
    LOG(ERROR) << "Memory pool is already set up. I cannot set up it twice.";
    return false;
  }

  // The actual initialization.
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  // Initialize the flags for the memory pool.
  memory_pool_available_for_device_.resize(device_count, false);
  per_device_streams_.resize(device_count, nullptr);
  // Push the current device so we can recover later.
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  vector<cnmemDevice_t> cnmem_devs(device_ids.size());
  for (int i = 0; i < device_ids.size(); ++i) {
    const int device_id = device_ids[i];
    CHECK_GE(device_id, 0);
    CHECK_LT(device_id, device_count);
    // This ensures we do not specify the same device twice.
    CHECK(!memory_pool_available_for_device_[device_id]);
    CUDA_CHECK(cudaSetDevice(device_id));
    size_t free_memory, used_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &used_memory));
    LOG(INFO) << "Reserving " << proportion_of_memory_to_reserve * 100
              << "percent of the free memory (total " << free_memory
              << ") on device " << device_id;
    // Note: we create a dummy non-null stream for memory allocations, so that
    // any malloc can be called from any cuda stream, since caffe2 uses a lot of
    // non-default streams for computation. We will allocate all the reserved
    // memory to that non-null stream.
    cnmem_devs[i].device = device_id;
    cnmem_devs[i].size = size_t(proportion_of_memory_to_reserve * free_memory);
    CUDA_CHECK(cudaStreamCreate(&per_device_streams_[i]));
    cnmem_devs[i].numStreams = 1;
    cnmem_devs[i].streams = &per_device_streams_[i];
    cnmem_devs[i].streamSizes = &cnmem_devs[i].size;
    memory_pool_available_for_device_[device_id] = true;
  }
  CNMEM_CHECK(
      cnmemInit(cnmem_devs.size(), cnmem_devs.data(), CNMEM_FLAGS_DEFAULT));
  // After initialization, let's set back the device.
  CUDA_CHECK(cudaSetDevice(initial_device));
  LOG(INFO) << "Set up memory pool.";
  is_memory_pool_setup_ = true;
  return true;
}

bool CudaMemoryPool::FinalizeMemoryPool() {
  // If it has not been set up yet, we have nothing to do.
  if (!is_memory_pool_setup_) {
    return true;
  }
  CNMEM_CHECK(cnmemFinalize());
  for (int i = 0; i < per_device_streams_.size(); ++i) {
    if (per_device_streams_[i]) {
      CUDA_CHECK(cudaStreamDestroy(per_device_streams_[i]));
    }
  }
  // Reset all the static variables
  per_device_streams_.resize(0);
  memory_pool_available_for_device_.resize(0);
  memory_allocated_before_setup_ = false;
  is_memory_pool_setup_ = false;
  return true;
}

void* CudaMemoryPool::NewWithMemoryPool(size_t nbytes) {
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  CHECK(memory_pool_available_for_device_[device_id])
      << "Trying to allocate on device " << device_id
      << ", but memory pool is not initialized on that device.";
  void* ptr;
  CNMEM_CHECK(cnmemMalloc(&ptr, nbytes, per_device_streams_[device_id]));
  return ptr;
}

void CudaMemoryPool::DeleteWithMemoryPool(void* data) {
  cudaPointerAttributes attr;
  CUDA_CHECK(cudaPointerGetAttributes(&attr, data));
  DCHECK_EQ(attr.memoryType, cudaMemoryTypeDevice);
  CHECK(memory_pool_available_for_device_[attr.device])
      << "Current pointer belongs to " << attr.device
      << ", but memory pool is not initialized on that device. "
      << "Was your pointer allocated using the memory pool?";
  CNMEM_CHECK(cnmemFree(data, per_device_streams_[attr.device]));
}

}  // namespace caffe2