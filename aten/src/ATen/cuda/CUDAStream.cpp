#include "ATen/cuda/CUDAStream.h"
#include "ATen/DeviceGuard.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/CUDAEvent.h"
#include "ATen/cuda/Exceptions.h"
#include "c10/util/Exception.h"

#include <mutex>
#include <atomic>
#include <cstdint>
#include <deque>
#include <vector>
#include <array>

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
static int64_t num_gpus = -1;
static constexpr int kStreamsPerPool = 32;
static constexpr unsigned int kDefaultFlags = cudaStreamNonBlocking;

// Note: stream priority is not supported by HIP
// Note: lower numbers are higher priorities, zero is default priority
#ifndef __HIP_PLATFORM_HCC__
  static int kHighPriority = -1;
  static int kLowPriority = 0;
#endif // __HIP_PLATFORM_HCC__

// Default streams
static std::once_flag init_flag;
static std::vector<CUDAStreamInternals> default_streams;

// Non-default streams
// Note: the number of CUDA devices is determined at run time,
// and the low and high priority pools are lazily initialized 
// when the first stream is requested for a device.
// The device flags track the initialization of each device, while
// the low and high priority counters track, for each device, the next stream 
// in the pool to be returned when a stream is requested (round-robin fashion
// , see the note in CUDAStream.h). 
static std::deque<std::once_flag> device_flags;
static std::deque<std::atomic<uint32_t>> low_priority_counters;
static std::deque<std::atomic<uint32_t>> high_priority_counters;
static std::vector<std::array<CUDAStreamInternals, kStreamsPerPool>> low_priority_streams;
static std::vector<std::array<CUDAStreamInternals, kStreamsPerPool>> high_priority_streams;

// Thread-local current streams
static thread_local CUDAStreamInternals** current_streams = nullptr;

// Populates global values and creates a default stream for each device.
// Note: the default stream on each device is signified by a nullptr, 
// and so is not created as usual.
// In particular, we don't need to switch devices when creating the
// streams.
// Warning: this function must only be called once!
static void initGlobalStreamState() {
  num_gpus = getNumGPUs();

  // Resizes deques and vectors
  default_streams.resize(num_gpus);
  device_flags.resize(num_gpus);
  low_priority_counters.resize(num_gpus);
  high_priority_counters.resize(num_gpus);
  low_priority_streams.resize(num_gpus);
  high_priority_streams.resize(num_gpus);

  // Initializes default streams
  for (auto i = decltype(num_gpus){0}; i < num_gpus; ++i) {
    default_streams[i].device = i;
    low_priority_counters[i] = 0;
    high_priority_counters[i] = 0;
  }
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
static void initDeviceStreamState(const int64_t device) {
  // Switches to the requested device so streams are properly associated
  // with it.
  at::DeviceGuard device_guard{(int)device};
  
  for (auto i = decltype(kStreamsPerPool){0}; i < kStreamsPerPool; ++i) {
    auto& lowpri_stream = low_priority_streams[device][i];
    auto& hipri_stream = high_priority_streams[device][i];
    
    lowpri_stream.device = device;
    hipri_stream.device = device;
    
    #ifndef __HIP_PLATFORM_HCC__
      AT_CUDA_CHECK(cudaStreamCreateWithPriority(
        &lowpri_stream.stream
      , kDefaultFlags
      , kLowPriority));
      AT_CUDA_CHECK(cudaStreamCreateWithPriority(
        &hipri_stream.stream
      , kDefaultFlags
      , kHighPriority));
    #else
      AT_CUDA_CHECK(cudaStreamCreateWithFlags(
        &lowpri_stream.stream
      , kDefaultFlags));
      AT_CUDA_CHECK(cudaStreamCreateWithFlags(
        &hipri_stream.stream
      , kDefaultFlags));
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

CUDAStreamInternals* CUDAStream_getDefaultStream(int64_t device) {
  initCUDAStreamsOnce();
  if (device == -1) device = current_device();
  check_gpu(device);
  return &default_streams[device];
}

// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in CUDAStream.h)
static uint32_t get_idx(std::atomic<uint32_t> &counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
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
  
  // Initializes the stream pools (once)
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

void CUDAStream_synchronize_with(CUDAStreamInternals* ptr, const CUDAEvent& event) {
    if (event.isCreated())
      AT_CUDA_CHECK(cudaStreamWaitEvent(ptr->stream, event, 0));
}

} // namespace detail

void CUDAStream::synchronize_with(const CUDAEvent& event) const {
    detail::CUDAStream_synchronize_with(internals_, event);
}

} // namespace cuda
} // namespace at
