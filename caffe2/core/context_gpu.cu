#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>
#include <unordered_map>

#include "cub/util_allocator.cuh"
#include "cnmem.h"

#include "caffe2/core/asan.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/string_utils.h"

#define CNMEM_CHECK(condition)                                                 \
  do {                                                                         \
    cnmemStatus_t error = condition;                                           \
    CAFFE_ENFORCE_EQ(error, CNMEM_STATUS_SUCCESS, cnmemGetErrorString(error)); \
  } while (0)

CAFFE2_DEFINE_string(caffe2_cuda_memory_pool, "",
              "Sets the memory pool used by caffe2. Possible values are "
              "none, cnmen and cub.");
CAFFE2_DEFINE_double(caffe2_cnmem_reserve, 0.8,
             "Sets the proportion of memory pre-allocated by the memory "
             "pool if you use cnmem.");
CAFFE2_DEFINE_string(caffe2_cnmem_gpus, "",
              "A comma separated list containing the index of gpus that "
              "we will set the memory pool on. If not set, we will set "
              "up the memory pool on all available GPUs. This only applies "
              "to cnmem.");
// TODO(jiayq): Figure out the best default values for the params below.
// Currently we are using the setting copied from caffe.
CAFFE2_DEFINE_int(caffe2_cub_bin_growth, 2,
             "If using cub as the memory allocator, sets the growth of bins "
             "used by the cub pool.");
CAFFE2_DEFINE_int(caffe2_cub_min_bin, 6,
             "If using cub as the memory allocator, sets the min number of "
             "bins.");
CAFFE2_DEFINE_int(caffe2_cub_max_bin, 16,
             "If using cub as the memory allocator, sets the max number of "
             "bins.");

namespace caffe2 {

CAFFE_KNOWN_TYPE(Tensor<CUDAContext>);

thread_local ThreadLocalCUDAObjects CUDAContext::cuda_objects_;

// TODO(jiayq): these variables shouldn't be currently accessed during static
// initialization. We should consider moving them to a Mayer's singleton to
// be totally safe against SIOF.

// Static global variables for setting up the memory pool.
CudaMemoryPoolType g_cuda_memory_pool_type;
// For cnmem allocator
vector<bool> g_cnmem_available_for_device;
// For cub allocator
unique_ptr<cub::CachingDeviceAllocator> g_cub_allocator;
// an unordered map that holds the map from the cuda memory pointer to the
// device id that it is allocated from. This is used in the cuda memory pool
// cases, where we need the device id to carry out the deletion.
// Note(jiayq): an alternate approach is to use cudaGetPointerAttributes, but
// that is usually quite slow. We might want to benchmark the speed difference
// though.
// Note(jiayq): another alternate approach is to augment the Tensor class that
// would allow one to record the device id. However, this does not address any
// non-tensor allocation and deallocation.
// Ideally, a memory pool should already have the device id information, as
// long as we are using UVA (as of CUDA 5 and later) so the addresses are
// unique.
static std::unordered_map<void*, uint8_t> g_cuda_device_affiliation;

CudaMemoryPoolType GetCudaMemoryPoolType() {
  return g_cuda_memory_pool_type;
}

///////////////////////////////////////////////////////////////////////////////
// A wrapper to allow us to lazily initialize all cuda environments that Caffe
// uses. This gets done the first time a caffe2::CUDAContext::New() gets called
// which is probably the decisive indication that this caffe2 run is going to
// use GPUs. We avoid cuda initialization with core/init.h functionalities so
// that we have minimal resource impact in case we will need to run multiple
// caffe2 instances on a GPU machine.
///////////////////////////////////////////////////////////////////////////////

static void Caffe2InitializeCuda() {
  // If the current run does not have any cuda devices, do nothing.
  if (!HasCudaGPU()) {
    VLOG(1) << "No cuda gpu present. Skipping.";
    return;
  }
  // Check if the number of GPUs matches the expected compile-time max number
  // of GPUs.
  CAFFE_ENFORCE_LE(
      NumCudaDevices(),
      CAFFE2_COMPILE_TIME_MAX_GPUS,
      "Number of CUDA devices on the machine is larger than the compiled "
      "max number of gpus expected (",
      CAFFE2_COMPILE_TIME_MAX_GPUS,
      "). Increase that and recompile the caffe binary.");
  // Save the current device so we can restore it after moving across
  // different devices.
  int init_device;
  CUDA_CHECK(cudaGetDevice(&init_device));

  for (int i = 0; i < NumCudaDevices(); ++i) {
    auto err = cudaSetDevice(i);
    if (err != cudaSuccess) {
      LOG(WARNING)
          << "Cannot use device " << i
          << "due to the following error: " << cudaGetErrorString(err);
      continue;
    }
    // Enable peer access.
    for (int j = 0; j < NumCudaDevices(); ++j) {
      if (i == j) continue;
      int can_access;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
      if (can_access) {
        VLOG(1) << "Enabling peer access from " << i << " to " << j;
        // Note: just for future reference, the 0 here is not a gpu id, it is
        // a reserved flag for cudaDeviceEnablePeerAccess that should always be
        // zero currently.
        CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
      }
    }
  }
  // Restore the current device.
  CUDA_CHECK(cudaSetDevice(init_device));
  
  RegisterShapeCallFunction(
    TypeMeta::Id<Tensor<CUDAContext>>(),
    GetTensorShape<CUDAContext>
  );
}

static void SetUpCNMEM() {
  g_cnmem_available_for_device.assign(NumCudaDevices(), false);
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

static void Caffe2SetCUDAMemoryPool() {
  if (FLAGS_caffe2_cuda_memory_pool == "" ||
      FLAGS_caffe2_cuda_memory_pool == "none") {
    g_cuda_memory_pool_type = CudaMemoryPoolType::NONE;
  } else if (FLAGS_caffe2_cuda_memory_pool == "cnmem") {
    // sets up cnmem.
    g_cuda_memory_pool_type = CudaMemoryPoolType::CNMEM;
    SetUpCNMEM();
  } else if (FLAGS_caffe2_cuda_memory_pool == "cub") {
    // Sets up cub.
    g_cuda_memory_pool_type = CudaMemoryPoolType::CUB;
    SetUpCub();
  } else {
    CAFFE_THROW("Unrecognized cuda memory pool type: ",
                FLAGS_caffe2_cuda_memory_pool);
  }
}

// An initialization function that sets the CPU side to use pinned cpu
// allocator.
void Caffe2UsePinnedCPUAllocator() {
#if CAFFE2_ASAN_ENABLED
  // Note(jiayq): for more details, see
  //     https://github.com/google/sanitizers/issues/629
  LOG(WARNING) << "There are known issues between address sanitizer and "
                  "cudaMallocHost. As a result, caffe2 will not enable pinned "
                  "memory allocation in asan mode. If you are expecting any "
                  "behavior that depends on asan, be advised that it is not "
                  "turned on.";
#else
  if (!HasCudaGPU()) {
    VLOG(1) << "No GPU present. I won't use pinned allocator then.";
    return;
  }
  VLOG(1) << "Caffe2 gpu: setting CPUAllocator to PinnedCPUAllocator.";
  SetCPUAllocator(new PinnedCPUAllocator());
#endif
}

// Caffe2CudaInitializerHelper is a minimal struct whose sole purpose is to
// detect the first hint that this Caffe2 run is going to use GPU: either
// CUDAContext is initialized or CUDAContext::New is called. It then runs
// all the related cuda initialization functions.
namespace {
struct Caffe2CudaInitializerHelper {
  Caffe2CudaInitializerHelper() {
    // We cannot use bool because nvcc changes bool to __nv_bool which does
    // not have a std::atomic instantiation.
    static std::atomic<char> first_call(1);
    if (first_call.fetch_and((char)0)) {
      Caffe2InitializeCuda();
      Caffe2SetCUDAMemoryPool();
      Caffe2UsePinnedCPUAllocator();
    }
  }
};
}  // namespace

CUDAContext::CUDAContext(const int gpu_id)
    : gpu_id_(gpu_id == -1 ? GetDefaultGPUID() : gpu_id)
    , random_seed_(math::randomNumberSeed()) {
  static Caffe2CudaInitializerHelper g_cuda_initializer_;
}

CUDAContext::CUDAContext(const DeviceOption& option)
    : gpu_id_(option.has_cuda_gpu_id() ?
              option.cuda_gpu_id() : GetDefaultGPUID()),
      random_seed_(option.has_random_seed() ?
                   option.random_seed() : math::randomNumberSeed()) {
  static Caffe2CudaInitializerHelper g_cuda_initializer_;
  DCHECK_EQ(option.device_type(), CUDA);
}

// shared mutex to lock out alloc / free during NCCL launches
std::mutex& CUDAContext::mutex() {
  static std::mutex m;
  return m;
}

void* CUDAContext::New(size_t nbytes) {
  // Lock the mutex
  std::lock_guard<std::mutex> lock(CUDAContext::mutex());
  // A one-time caffe2 cuda initializer.
  static Caffe2CudaInitializerHelper g_cuda_initializer_;
  void* ptr = nullptr;
  switch (g_cuda_memory_pool_type) {
  case CudaMemoryPoolType::NONE:
    CUDA_CHECK(cudaMalloc(&ptr, nbytes));
    return ptr;
  case CudaMemoryPoolType::CNMEM: {
    auto gpuId = GetCurrentGPUID();
    CAFFE_ENFORCE(
        gpuId < g_cnmem_available_for_device.size() &&
            g_cnmem_available_for_device[gpuId],
        "Trying to allocate on device ",
        gpuId,
        " but cnmem pool is not set up for it.");
    CNMEM_CHECK(cnmemMalloc(&ptr, nbytes, nullptr));
    g_cuda_device_affiliation[ptr] = GetCurrentGPUID();
    VLOG(2) << "CNMEM allocating pointer " << ptr << " on device "
            << GetCurrentGPUID();
    return ptr;
  }
  case CudaMemoryPoolType::CUB:
    CUDA_CHECK(g_cub_allocator->DeviceAllocate(&ptr, nbytes));
    g_cuda_device_affiliation[ptr] = GetCurrentGPUID();
    VLOG(2) << "CUB allocating pointer " << ptr << " on device "
            << GetCurrentGPUID();
    return ptr;
  }
  return nullptr;
}

void CUDAContext::Delete(void* ptr) {
  // lock the mutex
  std::lock_guard<std::mutex> lock(CUDAContext::mutex());

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
  case CudaMemoryPoolType::CNMEM: {
    auto it = g_cuda_device_affiliation.find(ptr);
    DCHECK(it != g_cuda_device_affiliation.end());
    DeviceGuard guard(it->second);
    VLOG(2) << "CNMEM freeing pointer " << ptr << " on device " << it->second;
    CNMEM_CHECK(cnmemFree(ptr, nullptr));
    g_cuda_device_affiliation.erase(it);
    break;
  }
  case CudaMemoryPoolType::CUB: {
    auto it = g_cuda_device_affiliation.find(ptr);
    DCHECK(it != g_cuda_device_affiliation.end());
    VLOG(2) << "CUB freeing pointer " << ptr << " on device " << it->second;
    CUDA_CHECK(g_cub_allocator->DeviceFree(it->second, ptr));
    g_cuda_device_affiliation.erase(it);
    break;
  }
  }
}

}  // namespace caffe2
