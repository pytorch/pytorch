#include "ATen/cuda/CUDAStream.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/CUDAEvent.h"
#include "ATen/cuda/Exceptions.h"
#include "ATen/core/Error.h"

#include <mutex>
#include <atomic>

// Internal implementation is entirely hidden
struct CUDAStreamInternals {
  bool is_destructible;
  std::atomic<int> refcount;
  int64_t device; // Note: cudaGetDevice works with int32_t, not int64_t
  cudaStream_t stream;
};

namespace at {
namespace cuda {

namespace detail {

  /*
  * Stream state
  */
  static constexpr cudaStream_t DEFAULT_STREAM = 0;

  static std::once_flag init_flag;
  static int64_t num_gpus;
  static CUDAStreamInternals* default_streams;
  static thread_local CUDAStreamInternals** current_streams = nullptr;

  // Creates a(n indestructible) default stream for each device
  // Note: the default stream on each device is signified by a zero
  // value for the pointer, and so is not actually created as usual.
  // In particular, we don't need to switch devices when creating the
  // streams.
  static void initDefaultCUDAStreams() {
    num_gpus = getNumGPUs();
    default_streams = (CUDAStreamInternals*) malloc(num_gpus * sizeof(CUDAStreamInternals));
    for (auto i = decltype(num_gpus){0}; i < num_gpus; ++i) {
      default_streams[i].is_destructible = false;
      default_streams[i].refcount = 0;
      default_streams[i].device = i;
      default_streams[i].stream = DEFAULT_STREAM;
    }
  }

  // Init front-end to ensure initialization only occurs once
  static void initCUDAStreamsOnce() {
    // Inits default streams (once, globally)
    std::call_once(init_flag, initDefaultCUDAStreams);

    // Inits current streams (thread local) to default streams
    if (current_streams) return;
    current_streams = (CUDAStreamInternals**) malloc(num_gpus * sizeof(CUDAStreamInternals*));
    for (auto i = decltype(num_gpus){0}; i < num_gpus; ++i) {
      current_streams[i] = &default_streams[i];
    }
  }

  /*
  * Pointer-based stream API
  */

  // Helper to verify the GPU index is valid
  static inline void check_gpu(int64_t device) {
    AT_ASSERT(device >= 0 && device < num_gpus);
  }

  CUDAStreamInternals* CUDAStream_getDefaultStreamOnDevice(int64_t device) {
    initCUDAStreamsOnce();
    check_gpu(device);
    return &default_streams[device];
  }
  CUDAStreamInternals* CUDAStream_getDefaultStream() {
    return CUDAStream_getDefaultStreamOnDevice(current_device());
  }

  // Creates (and retains) and new cuda stream
  CUDAStreamInternals* CUDAStream_createAndRetainWithOptions(int32_t flags, int32_t priority) {
    CUDAStreamInternals* internals = (CUDAStreamInternals*) malloc(sizeof(CUDAStreamInternals));
    internals->is_destructible = true;
    internals->refcount = 1;
    internals->device = current_device();
    #ifndef __HIP_PLATFORM_HCC__
      AT_CUDA_CHECK(cudaStreamCreateWithPriority(&internals->stream, flags, priority));
    #else
      AT_CUDA_CHECK(cudaStreamCreateWithFlags(&internals->stream, flags));
    #endif // __HIP_PLATFORM_HCC__
    return internals;
  }

  // Note: despite not being "unsafe," is using these methods in a multithreaded
  // environment then the caller must be sure that streams are valid
  // when they're requested. These methods will throw an error if an
  // invalid stream is requested.
  CUDAStreamInternals* CUDAStream_getAndRetainCurrentStreamOnDevice(int64_t device) {
    initCUDAStreamsOnce();
    check_gpu(device);
    auto cur = current_streams[device];
    AT_ASSERT(CUDAStream_retain(cur));
    return cur;
  }
  CUDAStreamInternals* CUDAStream_getAndRetainCurrentStream() {
    return CUDAStream_getAndRetainCurrentStreamOnDevice(current_device());
  }

  // Note: these unsafe methods do not retain the stream before returning it.
  // This is unsafe behavior and these methods SHOULD NOT BE USED.
  // They are here only for legacy compatibility.
  CUDAStreamInternals* CUDAStream_getCurrentStreamOnDeviceUnsafe(int64_t device) {
    initCUDAStreamsOnce();
    check_gpu(device);
    return current_streams[device];
  }
  CUDAStreamInternals* CUDAStream_getCurrentStreamUnsafe() {
    return CUDAStream_getCurrentStreamOnDeviceUnsafe(current_device());
  }

  void CUDAStream_setStreamOnDevice(int64_t device, CUDAStreamInternals* ptr) {
    initCUDAStreamsOnce();
    check_gpu(device);
    AT_ASSERT(ptr);
    AT_ASSERT(ptr->device == device);
    AT_ASSERT(CUDAStream_retain(ptr));

    CUDAStream_free(current_streams[device]);
    current_streams[device] = ptr;
  }

  void CUDAStream_uncheckedSetStreamOnDevice(int64_t device, CUDAStreamInternals* ptr) {
    initCUDAStreamsOnce();
    CUDAStream_uncheckedFree(current_streams[device]);
    current_streams[device] = ptr;
  }

  void CUDAStream_setStream(CUDAStreamInternals* ptr) {
    CUDAStream_setStreamOnDevice(current_device(), ptr);
  }

  // Getters
  cudaStream_t CUDAStream_stream(CUDAStreamInternals* ptr) {
    AT_ASSERT(ptr);
    return ptr->stream;
  }

  int64_t CUDAStream_device(CUDAStreamInternals* ptr) {
    AT_ASSERT(ptr);
    return ptr->device;
  }

  // Memory management
  // Note: only destructible (non-default) streams are ref counted
  bool CUDAStream_retain(CUDAStreamInternals* ptr) {
    AT_ASSERT(ptr);
    if (ptr->is_destructible) return(++ptr->refcount > 1);
    return true;
  }

  void CUDAStream_free(CUDAStreamInternals*& ptr) {
    if (ptr && ptr->stream && ptr->is_destructible && --ptr->refcount <= 0) {
      AT_ASSERT(ptr->refcount == 0);
      AT_CUDA_CHECK(cudaStreamDestroy(ptr->stream));
      free(ptr);
      ptr = nullptr;
    }
  }
  void CUDAStream_uncheckedFree(CUDAStreamInternals*& ptr) {
    if (ptr && ptr->stream && ptr->is_destructible && --ptr->refcount <= 0) {
      cudaStreamDestroy(ptr->stream);
      free(ptr);
      ptr = nullptr;
    }
  }

  void CUDAStream_synchronize_with(CUDAStreamInternals* ptr, const CUDAEvent& event) {
    AT_CUDA_CHECK(cudaStreamWaitEvent(ptr->stream, event, 0));
  }

} // namespace detail

  /*
  * CUDAStream functions
  */

   // Copy constructor
  CUDAStream::CUDAStream(const CUDAStream& other) {
    AT_ASSERT(other.internals_);
    AT_ASSERT(detail::CUDAStream_retain(other.internals_));

    internals_ = other.internals_;
  }

  // Move constructor
  CUDAStream::CUDAStream(CUDAStream&& other) {
    AT_ASSERT(other.internals_);

    std::swap(internals_, other.internals_);
  }

  void CUDAStream::synchronize_with(const CUDAEvent& event) const {
    detail::CUDAStream_synchronize_with(internals_, event);
  }

} // namespace cuda
} // namespace at
