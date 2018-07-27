#include "ATen/cuda/CUDAStream.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/Exceptions.h"
#include "ATen/Error.h"

#include <mutex>
#include <atomic>
#include <cstdint>
#include <deque>
#include <vector>

// Internal implementation is entirely hidden
// Note: CUDAStreamInternals doubles for a THCStream
struct CUDAStreamInternals {
  CUDAStreamInternals() = default;

  ~CUDAStreamInternals() {
    if (stream) cudaStreamDestroy(stream);
  }

  int64_t device = -1; 
  cudaStream_t stream = nullptr;
};

namespace at {
namespace cuda {

namespace detail {

  // Global stream state and constants
  static int64_t num_gpus;
  static constexpr int STREAMS_PER_POOL = 32;
  static constexpr unsigned int DEFAULT_FLAGS = cudaStreamNonBlocking;
  static int HIGH_PRIORITY = 0;
  static int LOW_PRIORITY = 0;
  
  // Default streams
  static std::once_flag init_flag;
  static std::vector<CUDAStreamInternals> default_streams;

  // Non-default streams
  static std::deque<std::once_flag> device_flags;
  static std::deque<std::atomic<int>> low_priority_counters;
  static std::deque<std::atomic<int>> high_priority_counters;
  static std::vector<std::vector<CUDAStreamInternals>> low_priority_streams;
  static std::vector<std::vector<CUDAStreamInternals>> high_priority_streams;

  // Thread-local current streams
  static thread_local CUDAStreamInternals** current_streams = nullptr;

  // Populates global values and creates a default stream for each device.
  // Note: the default stream on each device is signified by a nullptr, 
  // and so is not created as usual.
  // In particular, we don't need to switch devices when creating the
  // streams.
  static void initGlobalStreamState() {
    num_gpus = getNumGPUs();

    // Resizes deques and vectors
    default_streams.resize(num_gpus);
    device_flags.resize(num_gpus);
    low_priority_counters.resize(num_gpus);
    high_priority_counters.resize(num_gpus);
    low_priority_streams.resize(num_gpus);
    high_priority_streams.resize(num_gpus);

    // Initializes streams and counters
    for (auto i = decltype(num_gpus){0}; i < num_gpus; ++i) {
      default_streams[i].device = i;
      low_priority_counters[i] = 0;
      high_priority_counters[i] = 0;
    }

    // Populates low and high priority range
    AT_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&LOW_PRIORITY, &HIGH_PRIORITY));
  }

  // Creates the low and high priority stream pools for the specified device
  static void initDeviceStreamState(const int64_t device) {
    low_priority_streams[device].resize(STREAMS_PER_POOL);
    high_priority_streams[device].resize(STREAMS_PER_POOL);
    
    for (auto i = decltype(STREAMS_PER_POOL){0}; i < STREAMS_PER_POOL; ++i) {
      auto& lowpri_stream = low_priority_streams[device][i];
      auto& hipri_stream = high_priority_streams[device][i];
      
      lowpri_stream.device = device;
      hipri_stream.device = device;
      
      #ifndef __HIP_PLATFORM_HCC__
        AT_CUDA_CHECK(cudaStreamCreateWithPriority(
          &lowpri_stream.stream
        , DEFAULT_FLAGS
        , LOW_PRIORITY));
        AT_CUDA_CHECK(cudaStreamCreateWithPriority(
          &hipri_stream.stream
        , DEFAULT_FLAGS
        , HIGH_PRIORITY));
      #else 
        AT_CUDA_CHECK(cudaStreamCreateWithFlags(
          &lowpri_stream.stream
        , DEFAULT_FLAGS));
        AT_CUDA_CHECK(cudaStreamCreateWithFlags(
          &hipri_stream.stream
        , DEFAULT_FLAGS));
      #endif // __HIP_PLATFORM_HCC__
    }
  }

  // Init front-end to ensure initialization only occurs once
  static void initCUDAStreamsOnce() {
    // Inits default streams (once, globally)
    std::call_once(init_flag, initGlobalStreamState);

    if (current_streams) return;

    // Inits current streams (thread local) to default streams
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

  // Helper to determine the index of the stream to return
  // Streams are returned round-robin, and the counter is "kept" between
  // 0 and STREAMS_PER_POOl. 
  // Note: it is possible the counter will grow beyond STREAMS_PER_POOL 
  // temporarily, but it will eventually return to the desired range.
  static int get_idx(std::atomic<int> &counter) {
    int raw_idx = counter++;
    int modded = raw_idx % STREAMS_PER_POOL;
    if (raw_idx >= STREAMS_PER_POOL && modded == 0) {
      counter -= STREAMS_PER_POOL;
    }
    return modded;
  }

  CUDAStreamInternals* CUDAStream_getDefaultStream(int64_t device) {
    initCUDAStreamsOnce();
    if (device == -1) device = current_device();
    check_gpu(device);
    return &default_streams[device];
  }

  // Returns a stream from the requested pool
  // Note: when called the first time on a device, this will create the
  // stream pools for that device.
  CUDAStreamInternals* CUDAStream_createStream(
    const bool isHighPriority
  , int64_t device) {
    initCUDAStreamsOnce();
    if (device == -1) device = current_device();
    check_gpu(device);
    std::call_once(device_flags[device], initDeviceStreamState, device);

    if (isHighPriority) {
      const auto idx = get_idx(high_priority_counters[device]);
      return &high_priority_streams[device][idx];
    }

    const auto idx = get_idx(low_priority_counters[device]);
    return &low_priority_streams[device][idx];
}

  CUDAStreamInternals* CUDAStream_getCurrentStream(int64_t device) {
    initCUDAStreamsOnce();
    if (device == -1) device = current_device();
    check_gpu(device);
    return current_streams[device];
  }

  void CUDAStream_setStream(CUDAStreamInternals* ptr) {
    AT_ASSERT(ptr);
    current_streams[ptr->device] = ptr;
  }
  void CUDAStream_uncheckedSetStream(CUDAStreamInternals* ptr) {
    current_streams[ptr->device] = ptr;
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

} // namespace detail
} // namespace cuda
} // namespace at
