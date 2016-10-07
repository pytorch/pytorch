#include <algorithm>
#include <cstdlib>
#include <string>

#include "cub/util_allocator.cuh"
#include "cnmem.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/string_utils.h"


#define CNMEM_CHECK(condition) \
  do { \
    cnmemStatus_t error = condition; \
    CHECK_EQ(error, CNMEM_STATUS_SUCCESS) << cnmemGetErrorString(error); \
  } while (0)


DEFINE_string(caffe2_cuda_memory_pool, "",
              "Sets the memory pool used by caffe2. Possible values are "
              "none, cnmen and cub.");
DEFINE_double(caffe2_cnmem_reserve, 0.8,
             "Sets the proportion of memory pre-allocated by the memory "
             "pool if you use cnmem.");
DEFINE_string(caffe2_cnmem_gpus, "",
              "A comma separated list containing the index of gpus that "
              "we will set the memory pool on. If not set, we will set "
              "up the memory pool on all available GPUs. This only applies "
              "to cnmem.");
// TODO(jiayq): Figure out the best default values for the params below.
// Currently we are using the setting copied from caffe.
DEFINE_int32(caffe2_cub_bin_growth, 2,
             "If using cub as the memory allocator, sets the growth of bins "
             "used by the cub pool.");
DEFINE_int32(caffe2_cub_min_bin, 6,
             "If using cub as the memory allocator, sets the min number of "
             "bins.");
DEFINE_int32(caffe2_cub_max_bin, 16,
             "If using cub as the memory allocator, sets the max number of "
             "bins.");

namespace caffe2 {

CAFFE_KNOWN_TYPE(Tensor<CUDAContext>);

thread_local ThreadLocalCUDAObjects CUDAContext::cuda_objects_;

// Static global variables for setting up the memory pool.
CudaMemoryPoolType g_cuda_memory_pool_type;
bool g_memory_allocation_already_called = false;
// For cnmem allocator
vector<bool> g_cnmem_available_for_device(NumCudaDevices(), false);
// For cub allocator
unique_ptr<cub::CachingDeviceAllocator> g_cub_allocator;


CudaMemoryPoolType GetCudaMemoryPoolType() {
  return g_cuda_memory_pool_type;
}

void* CUDAContext::New(size_t nbytes) {
  g_memory_allocation_already_called = true;
  void* ptr = nullptr;
  switch (g_cuda_memory_pool_type) {
  case CudaMemoryPoolType::NONE:
    CUDA_CHECK(cudaMalloc(&ptr, nbytes));
    return ptr;
  case CudaMemoryPoolType::CNMEM:
    CAFFE_ENFORCE(
        g_cnmem_available_for_device[GetCurrentGPUID()],
        "Trying to allocate on device ", GetCurrentGPUID(),
        " but cnmem pool is not set up for it.");
    CNMEM_CHECK(cnmemMalloc(&ptr, nbytes, nullptr));
    return ptr;
  case CudaMemoryPoolType::CUB:
    CUDA_CHECK(g_cub_allocator->DeviceAllocate(&ptr, nbytes));
    return ptr;
  }
  return nullptr;
}

void CUDAContext::Delete(void* ptr) {
  switch (g_cuda_memory_pool_type) {
  case CudaMemoryPoolType::NONE: {
    // If memory pool is not set up, use simple cudaFree.
    cudaError_t error = cudaFree(ptr);
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
    break; }
  case CudaMemoryPoolType::CNMEM:
  	CNMEM_CHECK(cnmemFree(ptr, nullptr));
    break;
  case CudaMemoryPoolType::CUB:
    CUDA_CHECK(g_cub_allocator->DeviceFree(ptr));
    break;
  }
}

static void SetUpCNMEM() {
  VLOG(1) << "Setting up cnmem memory pool.";
  vector<int> device_ids;
  // If the cnmem gpus are not set, set up all gpus.
  if (FLAGS_caffe2_cnmem_gpus.size() == 0) {
    device_ids.resize(NumCudaDevices());
    for (int i = 0; i < device_ids.size(); ++i) {
      device_ids[i] = i;
    }
  } else {
    vector<string> device_ids_str = split(',', FLAGS_caffe2_cnmem_gpus);
    for (const string& id_str : device_ids_str) {
      int id = 0;
      try {
        id = std::stoi(id_str);
      } catch (...) {
        CAFFE_THROW(
            "Cannot parse device id ",
            id_str,
            " to a valid int number.");
      }
      device_ids.push_back(id);
    }
  }
  CAFFE_ENFORCE(FLAGS_caffe2_cnmem_reserve >= 0 &&
                FLAGS_caffe2_cnmem_reserve < 1.0,
                "caffe2_cnmem_reserve number must be in [0, 1)");
  vector<cnmemDevice_t> cnmem_devs(device_ids.size());
  for (int i = 0; i < device_ids.size(); ++i) {
    const int id = device_ids[i];
    CAFFE_ENFORCE(
        id >= 0 && id < NumCudaDevices(),
        "GPU id ", id, " out of the range of available GPUs.");
    DeviceGuard guard(id);
    size_t free, used;
    CUDA_CHECK(cudaMemGetInfo(&free, &used));
    VLOG(1) << "Reserving " << FLAGS_caffe2_cnmem_reserve * 100
            << " percent of the free memory (total " << free
            << ") on device " << id;
    // Note: we create a dummy non-null stream for memory allocations, so that
    // any malloc can be called from any cuda stream, since caffe2 uses a lot of
    // non-default streams for computation. We will allocate all the reserved
    // memory to that non-null stream.
    cnmem_devs[i].device = id;
    cnmem_devs[i].size = size_t(FLAGS_caffe2_cnmem_reserve * free);
    cnmem_devs[i].numStreams = 0;
    cnmem_devs[i].streamSizes = nullptr;
    g_cnmem_available_for_device[id] = true;
  }
  CNMEM_CHECK(
      cnmemInit(cnmem_devs.size(), cnmem_devs.data(), CNMEM_FLAGS_DEFAULT));
  VLOG(1) << "Done setting up cnmem memory pool.";
}

static void SetUpCub() {
  VLOG(1) << "Setting up cub memory pool.";
  const bool k_cub_debug =
  #ifdef NDEBUG
      false;
  #else
      true;
  #endif
  // Sets up the cub memory pool
  try {
    g_cub_allocator.reset(new cub::CachingDeviceAllocator(
        FLAGS_caffe2_cub_bin_growth,
        FLAGS_caffe2_cub_min_bin,
        FLAGS_caffe2_cub_max_bin,
        static_cast<size_t>(-1),
        false,
        k_cub_debug));
  } catch (...) {
    CAFFE_THROW("Some error happened at cub initialization.");
  }
  VLOG(1) << "Done setting up cub memory pool.";
}

// Global initializtion function to set up the cuda memory pool during
// construction time.
bool Caffe2SetCUDAMemoryPool(int*, char***) {
  if (!HasCudaGPU()) {
    VLOG(1) << "No GPU present. I won't set up cuda memory pool";
    return true;
  }
  if (g_memory_allocation_already_called) {
    LOG(ERROR) << "Caffe2SetCUDAMemoryPool should always be called before "
                  "any CUDAContext::New() calls are made.";
    return false;
  }
  if (FLAGS_caffe2_cuda_memory_pool == "" ||
      FLAGS_caffe2_cuda_memory_pool == "none") {
    g_cuda_memory_pool_type = CudaMemoryPoolType::NONE;
    return true;
  } else if (FLAGS_caffe2_cuda_memory_pool == "cnmem") {
    // sets up cnmem.
    g_cuda_memory_pool_type = CudaMemoryPoolType::CNMEM;
    SetUpCNMEM();
    return true;
  } else if (FLAGS_caffe2_cuda_memory_pool == "cub") {
    // Sets up cub.
    g_cuda_memory_pool_type = CudaMemoryPoolType::CUB;
    SetUpCub();
    return true;
  }
  LOG(ERROR) << "Unrecognized cuda memory pool type: "
             << FLAGS_caffe2_cuda_memory_pool;
  return false;
}

// An initialization function that sets the CPU side to use pinned cpu
// allocator.
bool Caffe2UsePinnedCPUAllocator(int*, char***) {
#ifdef __SANITIZE_ADDRESS__
  // Note(jiayq): for more details, see
  //     https://github.com/google/sanitizers/issues/629
  LOG(WARNING) << "There are known issues between address sanitizer and "
                  "cudaMallocHost. As a result, caffe2 will not enable pinned "
                  "memory allocation in asan mode. If you are expecting any "
                  "behavior that depends on asan, be advised that it is not "
                  "turned on.";
  return true;
#else
  if (!HasCudaGPU()) {
    VLOG(1) << "No GPU present. I won't use pinned allocator then.";
    return true;
  }
  VLOG(1) << "Caffe2 gpu: setting CPUAllocator to PinnedCPUAllocator.";
  SetCPUAllocator(new PinnedCPUAllocator());
  return true;
#endif
}

REGISTER_CAFFE2_INIT_FUNCTION(Caffe2SetCUDAMemoryPool,
                              &Caffe2SetCUDAMemoryPool,
                              "Sets up the cuda memory pool.");
REGISTER_CAFFE2_INIT_FUNCTION(Caffe2UsePinnedCPUAllocator,
                              &Caffe2UsePinnedCPUAllocator,
                              "Make the CPU side use pinned memory.");
}  // namespace caffe2
