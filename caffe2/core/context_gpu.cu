#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>
#include <unordered_map>

#include <ATen/Context.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "cub/util_allocator.cuh"

// Needed to be included first to check the CAFFE2_USE_CUDNN macros.
#include "caffe2/core/macros.h"

#include "caffe2/core/blob_stats.h"
#ifdef CAFFE2_USE_CUDNN
#include "caffe2/core/common_cudnn.h"
#endif // CAFFE2_USE_CUDNN
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/string_utils.h"
#include "caffe2/utils/cub_namespace.cuh"

C10_DEFINE_string(
    caffe2_cuda_memory_pool,
    "",
    "Sets the memory pool used by caffe2. Possible values are "
    "none, cnmem, thc and cub.");

// For description of CUB caching allocator configuration, see
// https://nvlabs.github.io/cub/structcub_1_1_caching_device_allocator.html
C10_DEFINE_int(
    caffe2_cub_bin_growth,
    8,
    "If using cub as the memory allocator, sets the growth of bins "
    "used by the cub pool.");
C10_DEFINE_int(
    caffe2_cub_min_bin,
    3,
    "If using cub as the memory allocator, sets the min number of "
    "bins.");
C10_DEFINE_int(
    caffe2_cub_max_bin,
    10,
    "If using cub as the memory allocator, sets the max number of "
    "bins.");
C10_DEFINE_int(
    caffe2_cub_max_managed_mb,
    10 * 1024,
    "If using cub as the memory allocators, sets the maximum amount "
    "of memory managed in gigabytes");

C10_DEFINE_bool(
    caffe2_cub_print_allocation_events,
    false,
    "If true CachingDeviceAllocator will print allocation and deallocation "
    "events to stdout.");

C10_DEFINE_bool(
    caffe2_gpu_memory_tracking,
    false,
    "If set, logs changes in GPU memory allocations");
C10_DEFINE_int(
    caffe2_gpu_memory_report_interval_mb,
    128,
    "The threshold in MB on how frequently to report memory changes");

namespace at {

REGISTER_CONTEXT(DeviceType::CUDA, caffe2::CUDAContext);
} // namespace at

namespace caffe2 {

// Generic implementation - CUDA will handle the right function to call for us
void CUDAContext::CopyBytesAsync(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device) {
  // TODO: verify that the CUDA handles copy from device to device correctly
  // even without SetDevice()
  // TODO: verify whether source or dest device should be a priority in picking
  // the stream
  // NB: right now the cross-device copy logic is invoked only in the contexts
  // when surrounding code explicitly manages data dependencies and sets up
  // events, so it's fine.  In order to make it a standalone function proper
  // synchronization between stream is required
  int gpu_id = 0;
  if (dst_device.is_cuda()) {
    gpu_id = dst_device.index();
  } else if (src_device.is_cuda()) {
    gpu_id = src_device.index();
  } else {
    LOG(FATAL) << "shouldn't be called with non-cuda device";
  }
  CUDA_ENFORCE(cudaMemcpyAsync(
      dst,
      src,
      nbytes,
      cudaMemcpyDefault,
      CUDAContext::getCudaObjects().GetStream(gpu_id)));
}

void CUDAContext::CopyBytesSync(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device) {
  // This emulates Caffe2 original behavior where sync copy doesn't change the
  // device. It's probably better for clarity to switch to the target device
  // explicitly here, but in the worst case CUDA would sync for us.
  // TODO: change it to CUDAGuard
  CUDAContext context(-1); // take current device
  CUDA_ENFORCE(cudaMemcpyAsync(
      dst, src, nbytes, cudaMemcpyDefault, context.cuda_stream()));
  // destructor of context synchronizes
}

// For the CPU context, we also allow a (probably expensive) function
// to copy the data from a cuda context. Inside the function, we create
// a temporary CUDAContext object to carry out the copy. From the caller's
// side, these functions are synchronous with respect to the host, similar
// to a normal CPUContext::CopyBytes<CPUContext, CPUContext> call.
template <>
inline void CPUContext::CopyBytes<CUDAContext, CPUContext>(
    size_t nbytes,
    const void* src,
    void* dst) {
  CUDAContext context(GetGPUIDForPointer(src));
  context.CopyBytes<CUDAContext, CPUContext>(nbytes, src, dst);
}
template <>
inline void CPUContext::CopyBytes<CPUContext, CUDAContext>(
    size_t nbytes,
    const void* src,
    void* dst) {
  CUDAContext context(GetGPUIDForPointer(dst));
  context.CopyBytes<CPUContext, CUDAContext>(nbytes, src, dst);
}

} // namespace caffe2

namespace caffe2 {

ThreadLocalCUDAObjects& CUDAContext::getCudaObjects() {
  static thread_local ThreadLocalCUDAObjects cuda_objects_;
  return cuda_objects_;
}

// TODO(jiayq): these variables shouldn't be currently accessed during static
// initialization. We should consider moving them to a Mayer's singleton to
// be totally safe against SIOF.

// Static global variables for setting up the memory pool.
CudaMemoryPoolType g_cuda_memory_pool_type;

std::unique_ptr<cub::CachingDeviceAllocator> g_cub_allocator;

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

// Data structures for optional memory tracking. Access to these structures
// is guarded by the CUDAContext::mutex.
static std::unordered_map<void*, long> g_size_map;
static std::vector<long> g_total_by_gpu_map(C10_COMPILE_TIME_MAX_GPUS, 0);
static std::vector<long> g_max_by_gpu_map(C10_COMPILE_TIME_MAX_GPUS, 0);

static long g_total_mem = 0;
static long g_last_rep = 0;

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
  C10_LOG_API_USAGE_ONCE("caffe2.init.cuda");
  // Check if the number of GPUs matches the expected compile-time max number
  // of GPUs.
  CAFFE_ENFORCE_LE(
      NumCudaDevices(),
      C10_COMPILE_TIME_MAX_GPUS,
      "Number of CUDA devices on the machine is larger than the compiled "
      "max number of gpus expected (",
      C10_COMPILE_TIME_MAX_GPUS,
      "). Increase that and recompile.");

  for (DeviceIndex i = 0; i < NumCudaDevices(); ++i) {
    CUDAGuard g(i);
    // Enable peer access.
    const int peer_group = i / CAFFE2_CUDA_MAX_PEER_SIZE;
    const int peer_start = peer_group * CAFFE2_CUDA_MAX_PEER_SIZE;
    const int peer_end = std::min(
        NumCudaDevices(), (peer_group + 1) * CAFFE2_CUDA_MAX_PEER_SIZE);
    VLOG(1) << "Enabling peer access within group #" << peer_group
            << ", from gpuid " << peer_start << " to " << peer_end - 1
            << ", for gpuid " << i << ".";

    for (int j = peer_start; j < peer_end; ++j) {
      if (i == j) continue;
      int can_access;
      CUDA_ENFORCE(cudaDeviceCanAccessPeer(&can_access, i, j));
      if (can_access) {
        VLOG(1) << "Enabling peer access from " << i << " to " << j;
        // Note: just for future reference, the 0 here is not a gpu id, it is
        // a reserved flag for cudaDeviceEnablePeerAccess that should always be
        // zero currently.
        // It is ok if peer access is already enabled...
        cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaDeviceEnablePeerAccess(j, 0));
        if ((err != cudaErrorPeerAccessAlreadyEnabled) &&
            (err != cudaSuccess)) {
          CAFFE_THROW(cudaGetErrorString(err));
        }
        cudaGetLastError(); // reset cuda error code
      }
    }
  }

#ifdef CAFFE2_USE_CUDNN
  // Check the versions of cuDNN that were compiled and linked with are compatible
  CheckCuDNNVersions();
#endif // CAFFE2_USE_CUDNN
}

static void SetUpCub() {
  VLOG(1) << "Setting up cub memory pool.";
  // Sets up the cub memory pool
  try {
    g_cub_allocator.reset(new cub::CachingDeviceAllocator(
        FLAGS_caffe2_cub_bin_growth,
        FLAGS_caffe2_cub_min_bin,
        FLAGS_caffe2_cub_max_bin,
        size_t(FLAGS_caffe2_cub_max_managed_mb) * 1024L * 1024L,
        false,
        FLAGS_caffe2_cub_print_allocation_events));
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
    CAFFE_THROW("CNMEM is no longer used by Caffe2. Use cub instead. "
                "This error message may go away in the future.");
  } else if (FLAGS_caffe2_cuda_memory_pool == "cub") {
    // Sets up cub.
    g_cuda_memory_pool_type = CudaMemoryPoolType::CUB;
    SetUpCub();
  } else if (FLAGS_caffe2_cuda_memory_pool == "thc") {
    g_cuda_memory_pool_type = CudaMemoryPoolType::THC;
    // Initialize caching allocator
    at::globalContext().lazyInitCUDA();
  } else {
    CAFFE_THROW(
        "Unrecognized cuda memory pool type: ", FLAGS_caffe2_cuda_memory_pool);
  }
}

/**
 * An allocator that does the CPU memory allocation with pinned memory.
 *
 * This is needed because if we want to do any asynchronous cuda memcpy,
 * the underlying CPU memory also needs to be allocated into pinned memory
 * space. As a result, whenever Caffe2 is built with GPU and there is
 * GPU present during runtime, at global initialization time we will set
 * the CPU memory allocator to allocate pinned memory.
 *
 * NB: This behavior is probably too aggressive. We should consider asking users
 * to do on-demand memory pinning (like exposed in PyTorch APIs) instead.
 */
struct CAFFE2_CUDA_API PinnedCPUAllocator final : public at::Allocator {
  PinnedCPUAllocator() {
    baseAllocator_ = GetDefaultCPUAllocator();
  }
  ~PinnedCPUAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    if (nbytes == 0) {
      // replicate c10::alloc_cpu behavior - return nullptr
      return {nullptr, nullptr, &Delete, at::Device(CPU)};
    }
    void* data;
    at::DataPtr data_ptr;
    std::lock_guard<std::mutex> lock(CUDAContext::mutex());
    if (IsNUMAEnabled()) {
      at::DeleterFnPtr expected_deleter = baseAllocator_->raw_deleter();
      data_ptr = baseAllocator_->allocate(nbytes);
      data = data_ptr.get();
      CAFFE_ENFORCE(data);
      CUDA_ENFORCE(cudaHostRegister(data, nbytes, cudaHostRegisterDefault));
      CAFFE_ENFORCE(
          data_ptr.compare_exchange_deleter(expected_deleter, &Delete),
          "Failed to swap deleter (already swapped?)");
    } else {
      CUDA_ENFORCE(cudaMallocHost(&data, nbytes));
      profiledCPUMemoryReporter().New(data, nbytes);
      data_ptr = {data, data, &Delete, at::Device(CPU)};
    }
    memset(data, 0, nbytes);
    return data_ptr;
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &Delete;
  }

 private:
  static void Delete(void* data) {
    if (!data) {
      return;
    }
    // Caffe2 uses a lazy way to figure out if one is actually going to use GPUs
    // or not. If a CUDAContext::New() call is made, inside the CUDAContext
    // function we will switch the cpu side allocator to a PinnedCPUAllocator.
    // But, if one calls CPUContext::New() before any cuda allocations,
    // PinnedCPUAllocator can still delete the corresponding memory.
    std::lock_guard<std::mutex> lock(CUDAContext::mutex());
    if (IsNUMAEnabled()) {
      CUDA_ENFORCE(cudaHostUnregister(data));
      GetDefaultCPUAllocator()->raw_deleter()(data);
    } else {
      cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaFreeHost(data));
      profiledCPUMemoryReporter().Delete(data);
      if (err == cudaErrorInvalidValue) {
        free(data);
        // Calling cudaGetLastError will reset the cuda error.
        cudaError_t _err = cudaGetLastError();
      } else {
        // For all other errors, still do a cuda check.
        CUDA_ENFORCE(err);
      }
    }
  }

  at::Allocator* baseAllocator_;
};

static PinnedCPUAllocator g_pinned_cpu_alloc;

// An initialization function that sets the CPU side to use pinned cpu
// allocator.
void Caffe2UsePinnedCPUAllocator() {
#if C10_ASAN_ENABLED
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

  // If CUDA is enabled, using CPU allocators other than PinnedCPUAllocator
  // will cause memory corruptions. Therefore, we need to set the priority
  // to highest to avoid being overwritten.
  SetCPUAllocator(
      &g_pinned_cpu_alloc,
      std::numeric_limits<uint8_t>::max() /* priority */);
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
} // namespace

/**
 * A utility function to rectify the gpu id. If the context specifies the
 * gpu id to be -1, it means that we will just use the current gpu id when
 * the function is being called.
 */
static inline DeviceIndex RectifyGPUID(DeviceIndex gpu_id) {
  return gpu_id == -1 ? CaffeCudaGetDevice() : gpu_id;
}

CUDAContext::CUDAContext(DeviceIndex gpu_id)
    : gpu_id_(RectifyGPUID(gpu_id)), random_seed_(RandomNumberSeed()) {
  static Caffe2CudaInitializerHelper g_cuda_initializer_;
}

CUDAContext::CUDAContext(const DeviceOption& option)
    : gpu_id_(
          option.has_device_id() ? RectifyGPUID(option.device_id())
                                   : CaffeCudaGetDevice()),
      random_seed_(
          option.has_random_seed() ? option.random_seed()
                                   : RandomNumberSeed()) {
  static Caffe2CudaInitializerHelper g_cuda_initializer_;
  TORCH_DCHECK_EQ(option.device_type(), PROTO_CUDA);
}

CUDAContext::~CUDAContext() {
  try {
    if (curand_generator_) {
      CURAND_CHECK(curandDestroyGenerator(curand_generator_));
    }
    // CUDAContext is used in 2 cases now:
    // - long-lived instance inside OperatorBase in which case what happens in
    //   destructor doesn't really matter
    // - short-lived on-the-fly instances that are utilized as CUDAGuard - in
    //   this case there's only one stream id (passed to SwitchToDevice) and
    //   it's preferrable to synchronize in the destructor
    FinishDeviceComputation();
  } catch (const std::exception& e)  {
    LOG(ERROR) << "Encountered following in " << __FUNCTION__ << ": " << e.what();
  }
}

// shared mutex to lock out alloc / free during NCCL launches
std::mutex& CUDAContext::mutex() {
  static std::mutex m;
  return m;
}

std::vector<long> CUDAContext::TotalMemoryByGpu() {
  std::lock_guard<std::mutex> lock(CUDAContext::mutex());
  CAFFE_ENFORCE(
      FLAGS_caffe2_gpu_memory_tracking,
      "Pass --caffe2_gpu_memory_tracking to enable memory stats");
  return g_total_by_gpu_map;
}

std::vector<long> CUDAContext::MaxMemoryByGpu() {
  std::lock_guard<std::mutex> lock(CUDAContext::mutex());
  CAFFE_ENFORCE(
      FLAGS_caffe2_gpu_memory_tracking,
      "Pass --caffe2_gpu_memory_tracking to enable memory stats");
  return g_max_by_gpu_map;
}

namespace {
void TrackMemoryAlloc(size_t nbytes) {
  int this_gpu = CaffeCudaGetDevice();
  g_total_by_gpu_map[this_gpu] += nbytes;
  g_max_by_gpu_map[this_gpu] =
      std::max(g_max_by_gpu_map[this_gpu], g_total_by_gpu_map[this_gpu]);
  g_total_mem += nbytes;
  if (g_total_mem - g_last_rep >
      FLAGS_caffe2_gpu_memory_report_interval_mb * 1024 * 1024) {
    for (int gpu = 0; gpu < g_total_by_gpu_map.size(); gpu++) {
      long t = g_total_by_gpu_map[gpu];
      long max_t = g_max_by_gpu_map[gpu];
      if (max_t > 0) {
        if (max_t != t) {
          VLOG(1) << "GPU " << gpu << ": " << t / 1024 / 1024 << " MB"
                  << " (max: " << max_t / 1024 / 1024 << " MB)";
        } else {
          VLOG(1) << "GPU " << gpu << ": " << t / 1024 / 1024 << " MB";
        }
      }
    }
    VLOG(1) << "Total: " << g_total_mem / 1024 / 1024 << " MB";
    g_last_rep = g_total_mem;
  }
}
}

struct DefaultCUDAAllocator final : public at::Allocator {
  DefaultCUDAAllocator() {}
  ~DefaultCUDAAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    // Lock the mutex
    std::lock_guard<std::mutex> lock(CUDAContext::mutex());
    // A one-time caffe2 cuda initializer.
    static Caffe2CudaInitializerHelper g_cuda_initializer_;
    void* ptr = nullptr;

    if (FLAGS_caffe2_gpu_memory_tracking) {
      TrackMemoryAlloc(nbytes);
    }
    switch (g_cuda_memory_pool_type) {
      case CudaMemoryPoolType::NONE:
        if (nbytes != 0) {
          CUDA_ENFORCE(cudaMalloc(&ptr, nbytes));
        }
        if (FLAGS_caffe2_gpu_memory_tracking) {
          g_size_map[ptr] = nbytes;
          g_cuda_device_affiliation[ptr] = CaffeCudaGetDevice();
        }
        return {ptr, ptr, &Delete, at::Device(CUDA, CaffeCudaGetDevice())};
      case CudaMemoryPoolType::CUB:
        if (nbytes != 0) {
          CUDA_ENFORCE(g_cub_allocator->DeviceAllocate(&ptr, nbytes));
        }
        g_cuda_device_affiliation[ptr] = CaffeCudaGetDevice();
        VLOG(2) << "CUB allocating pointer " << ptr << " on device "
                << CaffeCudaGetDevice();
        if (FLAGS_caffe2_gpu_memory_tracking) {
          g_size_map[ptr] = nbytes;
        }
        return {ptr, ptr, &Delete, at::Device(CUDA, CaffeCudaGetDevice())};
      case CudaMemoryPoolType::THC:
        {
          // The reason we have this stream guard here is to preserve
          // the historical behavior of the 'thc' allocator in Caffe2,
          // which is to put all allocations on the same (default)
          // stream.  This behavior is morally wrong (since passing
          // allocations between streams allows for the possibility
          // of you handing out some memory that an old stream
          // is still working on), but it doesn't seem to cause issues
          // in Caffe2 today.  Our hypothesis for why this is the case
          // is that Caffe2 doesn't really do very many allocations
          // on the fly; instead they allocate once and then reuse
          // the allocations for the whole program.  In this case,
          // the hazard is avoided.
          //
          // We intend to remove this stream guard, but the benefit
          // to putting all allocations on the same stream is it
          // reduces per-stream fragmentation, and this helps
          // some models that are currently running with the thc
          // allocator fit in memory.  We will need to find some
          // way of resolving this problem.
          cuda::CUDAStreamGuard g(
            Stream(
              Stream::DEFAULT,
              Device(kCUDA, CaffeCudaGetDevice())
            ));
          ptr = cuda::CUDACachingAllocator::raw_alloc(nbytes);
        }
        if (FLAGS_caffe2_gpu_memory_tracking) {
          g_size_map[ptr] = nbytes;
          g_cuda_device_affiliation[ptr] = CaffeCudaGetDevice();
        }
        return {ptr, ptr, &Delete, at::Device(CUDA, CaffeCudaGetDevice())};
    }
    return {nullptr, nullptr, &Delete, at::Device(CUDA, CaffeCudaGetDevice())};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &Delete;
  }

 private:
  static void Delete(void* ptr) {
    // lock the mutex
    std::lock_guard<std::mutex> lock(CUDAContext::mutex());
    if (FLAGS_caffe2_gpu_memory_tracking) {
      auto sz_it = g_size_map.find(ptr);
      DCHECK(sz_it != g_size_map.end());
      auto aff_it = g_cuda_device_affiliation.find(ptr);
      DCHECK(aff_it != g_cuda_device_affiliation.end());
      g_total_mem -= sz_it->second;
      g_total_by_gpu_map[aff_it->second] -= sz_it->second;
      g_size_map.erase(sz_it);
    }

    switch (g_cuda_memory_pool_type) {
      case CudaMemoryPoolType::NONE: {
        // If memory pool is not set up, use simple cudaFree.
        cudaError_t error = C10_CUDA_ERROR_HANDLED(cudaFree(ptr));
        // For some reason, in Python runtime we sometimes delete a data pointer
        // after the cuda runtime exits - this is odd but is probably caused by
        // a static workspace that pycaffe2 uses, and the destruction got
        // entangled in some race condition. Anyway, since cuda runtime is
        // exiting anyway, we will not need to worry about memory leak, so we
        // basically ignore it. This is definitely not ideal but works for now.
        if (error != cudaSuccess && error != cudaErrorCudartUnloading) {
          LOG(FATAL) << "Error at: " << __FILE__ << ":" << __LINE__ << ": "
                     << cudaGetErrorString(error);
        }

        if (FLAGS_caffe2_gpu_memory_tracking) {
          g_cuda_device_affiliation.erase(g_cuda_device_affiliation.find(ptr));
        }

        break;
      }
      case CudaMemoryPoolType::CUB: {
        auto it = g_cuda_device_affiliation.find(ptr);
        DCHECK(it != g_cuda_device_affiliation.end());
        VLOG(2) << "CUB freeing pointer " << ptr << " on device " << it->second;
        CUDA_ENFORCE(g_cub_allocator->DeviceFree(it->second, ptr));
        g_cuda_device_affiliation.erase(it);
        break;
      }
      case CudaMemoryPoolType::THC: {
        cuda::CUDACachingAllocator::raw_delete(ptr);
        if (FLAGS_caffe2_gpu_memory_tracking) {
          g_cuda_device_affiliation.erase(g_cuda_device_affiliation.find(ptr));
        }
        break;
      }
    }
  }
};

static DefaultCUDAAllocator g_cuda_alloc;
REGISTER_ALLOCATOR(CUDA, &g_cuda_alloc);

} // namespace caffe2

namespace at {
REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::CUDA,
    DeviceType::CUDA,
    caffe2::CUDAContext::CopyBytesSync,
    caffe2::CUDAContext::CopyBytesAsync);

REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::CUDA,
    DeviceType::CPU,
    caffe2::CUDAContext::CopyBytesSync,
    caffe2::CUDAContext::CopyBytesAsync);

REGISTER_COPY_BYTES_FUNCTION(
    DeviceType::CPU,
    DeviceType::CUDA,
    caffe2::CUDAContext::CopyBytesSync,
    caffe2::CUDAContext::CopyBytesAsync);
} // namespace at
