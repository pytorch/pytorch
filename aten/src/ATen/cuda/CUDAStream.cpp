#include "ATen/cuda/CUDAStream.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/CUDAEvent.h"
#include "ATen/cuda/CUDAGuard.h"
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
  int32_t stream_id = -1;
  cudaStream_t stream = nullptr;
};

namespace at {
namespace cuda {

namespace detail {

// Global stream state and constants
static int64_t num_gpus = -1;
static constexpr int kStreamsPerPoolBits = 5;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
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

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 2 bits --  -- 5 bits -----
// StreamIdType  stream id index
//  00 = default stream
//  01 = low priority stream
//  10 = high priority stream
//
// This is not really for efficiency; it's just easier to write the code
// to extract the index if we do this with bitmasks :)
//
// This is entirely an internal implementation detail, we reserve the right to
// renumber streams however we like.

enum class StreamIdType : uint8_t {
  DEFAULT = 0x0,
  LOW     = 0x1,
  HIGH    = 0x2,
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  switch (s) {
    case StreamIdType::DEFAULT:
      stream << "DEFAULT";
      break;
    case StreamIdType::LOW:
      stream << "LOW";
      break;
    case StreamIdType::HIGH:
      stream << "HIGH";
      break;
    default:
      stream << static_cast<uint8_t>(s);
      break;
  }
  return stream;
}

// Promotion makes me nervous...

static inline StreamIdType streamIdType(StreamId s) {
  return static_cast<StreamIdType>(s >> kStreamsPerPoolBits);
}

static inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(s & ((1 << kStreamsPerPoolBits) - 1));
}

StreamId makeStreamId(StreamIdType st, size_t si) {
  return (static_cast<StreamId>(st) << kStreamsPerPoolBits) | static_cast<StreamId>(si);
}

template <typename T>
static bool pointer_within(const T* ptr, ArrayRef<T> arr) {
  return std::greater_equal<const T*>()(ptr, arr.data()) && std::less<const T*>()(ptr, arr.data() + arr.size());
}

static StreamId CUDAStream_getStreamId(const CUDAStreamInternals* ptr) {
  // Hypothetically, we could store the stream ID in the stream.  But that
  // introduces a degree of freedom which could lead to bugs (where we
  // misnumber streams in the pool, or overwrite the number).  Better
  // to just compute it based on the metric that actually matters,
  // which is how we map IDs back into the vectors.

  DeviceIndex device_index = ptr->device;

  // Check if it's the default stream
  if (ptr == &default_streams[device_index]) {
    return makeStreamId(StreamIdType::DEFAULT, 0);
  }

  // Check if it's a low priority stream
  // NB: Because ptr may not necessarily lie within the array, we must use
  // std::less and similar templates to avoid UB that arises when
  // doing an operator< comparison.
  if (pointer_within<CUDAStreamInternals>(ptr, low_priority_streams[device_index])) {
    return makeStreamId(StreamIdType::LOW, ptr - low_priority_streams[device_index].data());
  }

  // Check if it's a high priority stream
  if (pointer_within<CUDAStreamInternals>(ptr, high_priority_streams[device_index])) {
    return makeStreamId(StreamIdType::HIGH, ptr - high_priority_streams[device_index].data());
  }

  AT_ASSERTM(0, "Could not compute stream ID for ", ptr, " on device ", device_index,
                " (something has gone horribly wrong!)");
}

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
  at::cuda::CUDAGuard device_guard{static_cast<int16_t>(device)};

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
CUDAStreamInternals* CUDAStream_getStreamFromPool(
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
cudaStream_t CUDAStream_stream(const CUDAStreamInternals* ptr) {
  AT_ASSERT(ptr);
  return ptr->stream;
}

int64_t CUDAStream_device(const CUDAStreamInternals* ptr) {
  AT_ASSERT(ptr);
  return ptr->device;
}

void CUDAStream_synchronize_with(const CUDAStreamInternals* ptr, const CUDAEvent& event) {
    if (event.isCreated())
      AT_CUDA_CHECK(cudaStreamWaitEvent(ptr->stream, event, 0));
}

} // namespace detail

CUDAStream::CUDAStream(const CUDAStreamInternals* ptr)
  : stream_(c10::Device(DeviceType::CUDA, detail::CUDAStream_device(ptr)), detail::CUDAStream_getStreamId(ptr)) {
}

void CUDAStream::synchronize_with(const CUDAEvent& event) const {
    detail::CUDAStream_synchronize_with(internals(), event);
}

// See Note [StreamId assignment]
CUDAStreamInternals* CUDAStream::internals() const {
  c10::DeviceIndex device_index = stream_.device_index();
  detail::StreamIdType st = detail::streamIdType(stream_.id());
  size_t si = detail::streamIdIndex(stream_.id());
  switch (st) {
    case detail::StreamIdType::DEFAULT:
      AT_ASSERTM(si == 0, "Unrecognized stream ", stream_,
                          " (I think this should be the default stream, but I got a non-zero index ", si, ")");
      return &detail::default_streams[device_index];
    case detail::StreamIdType::LOW:
      return &detail::low_priority_streams[device_index][si];
    case detail::StreamIdType::HIGH:
      return &detail::high_priority_streams[device_index][si];
    default:
      AT_ASSERTM(0, "Unrecognized stream ", stream_, " (I didn't recognize the stream type, ", st, ")");
  }
}

} // namespace cuda
} // namespace at
