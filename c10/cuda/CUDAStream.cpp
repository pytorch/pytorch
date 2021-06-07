#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace c10 {
namespace cuda {

namespace {

// Internal implementation that leaks the stream. It's not intended to be used
// outside of this file.
struct LeakyStreamInternals {
  LeakyStreamInternals() = default;
  C10_DISABLE_COPY_AND_ASSIGN(LeakyStreamInternals);

  ~LeakyStreamInternals() {
    // NB: this code is invoked only in the destruction of global variables
    // (since we never shrink the corresponding vectors). At this point the CUDA
    // runtime might be already destroyed and invoking cudaStreamDestroy leads
    // to a crash. It's likely an issue in CUDA, but to be safe - let's just
    // "forget" the destruction.

    // if (stream) cudaStreamDestroy(stream);
  }

  DeviceIndex device_index = -1;
  int32_t stream_id = -1;
  cudaStream_t stream = nullptr;
};

// Global stream state and constants
static DeviceIndex num_gpus = -1;
static constexpr int kStreamsPerPoolBits = 5;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
static constexpr unsigned int kDefaultFlags = cudaStreamNonBlocking;

// Note: lower numbers are higher priorities, zero is default priority
static int kHighPriority = -1;
static int kLowPriority = 0;

// Default streams
static std::once_flag init_flag;
static LeakyStreamInternals default_streams[C10_COMPILE_TIME_MAX_GPUS];

// Non-default streams
// Note: the number of CUDA devices is determined at run time,
// and the low and high priority pools are lazily initialized
// when the first stream is requested for a device.
// The device flags track the initialization of each device, while
// the low and high priority counters track, for each device, the next stream
// in the pool to be returned when a stream is requested (round-robin fashion
// , see the note in CUDAStream.h).
//
// unique_ptr<T[]> is used instead of vector<T> because T might be non-movable
// and non-copyable.
static std::once_flag device_flags[C10_COMPILE_TIME_MAX_GPUS];
static std::atomic<uint32_t> low_priority_counters[C10_COMPILE_TIME_MAX_GPUS];
static std::atomic<uint32_t> high_priority_counters[C10_COMPILE_TIME_MAX_GPUS];
static std::array<LeakyStreamInternals, kStreamsPerPool>
    low_priority_streams[C10_COMPILE_TIME_MAX_GPUS];
static std::array<LeakyStreamInternals, kStreamsPerPool>
    high_priority_streams[C10_COMPILE_TIME_MAX_GPUS];

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 25 bits -- -- 2 bits --  -- 5 bits -----
// zeros         StreamIdType  stream id index
//
// Where StreamIdType:
//  00 = default stream
//  01 = low priority stream
//  10 = high priority stream
//
// This is not really for efficiency; it's just easier to write the code
// to extract the index if we do this with bitmasks :)
//
// We are obligated to treat the stream ID 0 as the default stream, per the
// invariant specified in c10::Stream.  However, all other numbers are entirely
// an internal implementation detail, we reserve the right to renumber streams
// however we like.
//
// Note that it is really important that the MSB is zero; StreamId is a
// *signed* integer, and unsigned to signed conversion outside of the
// bounds of signed integer representation is undefined behavior.  You
// could work around this with something like
// https://stackoverflow.com/questions/13150449/efficient-unsigned-to-signed-cast-avoiding-implementation-defined-behavior
// but it seems a bit overkill for this.

enum class StreamIdType : uint8_t {
  DEFAULT = 0x0,
  LOW = 0x1,
  HIGH = 0x2,
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

// StreamId is 32-bit, so we can just rely on regular promotion rules.
// We rely on streamIdIndex and streamIdType being non-negative;
// see Note [Hazard when concatenating signed integers]

static inline StreamIdType streamIdType(StreamId s) {
  return static_cast<StreamIdType>(s >> kStreamsPerPoolBits);
}

static inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(s & ((1 << kStreamsPerPoolBits) - 1));
}

StreamId makeStreamId(StreamIdType st, size_t si) {
  return (static_cast<StreamId>(st) << kStreamsPerPoolBits) |
      static_cast<StreamId>(si);
}

template <typename T, typename A>
static bool pointer_within(const T* ptr, const A& arr) {
  return std::greater_equal<const T*>()(ptr, arr.data()) &&
      std::less<const T*>()(ptr, arr.data() + arr.size());
}

static StreamId CUDAStream_getStreamId(const LeakyStreamInternals* ptr) {
  // Hypothetically, we could store the stream ID in the stream.  But that
  // introduces a degree of freedom which could lead to bugs (where we
  // misnumber streams in the pool, or overwrite the number).  Better
  // to just compute it based on the metric that actually matters,
  // which is how we map IDs back into the vectors.

  DeviceIndex device_index = ptr->device_index;

  // Check if it's the default stream
  if (ptr == &default_streams[device_index]) {
    return makeStreamId(StreamIdType::DEFAULT, 0);
  }

  // Check if it's a low priority stream
  // NB: Because ptr may not necessarily lie within the array, we must use
  // std::less and similar templates to avoid UB that arises when
  // doing an operator< comparison.
  if (pointer_within<LeakyStreamInternals>(
          ptr, low_priority_streams[device_index])) {
    return makeStreamId(
        StreamIdType::LOW, ptr - low_priority_streams[device_index].data());
  }

  // Check if it's a high priority stream
  if (pointer_within<LeakyStreamInternals>(
          ptr, high_priority_streams[device_index])) {
    return makeStreamId(
        StreamIdType::HIGH, ptr - high_priority_streams[device_index].data());
  }

  TORCH_INTERNAL_ASSERT(
      0,
      "Could not compute stream ID for ",
      ptr,
      " on device ",
      device_index,
      " (something has gone horribly wrong!)");
}

// Thread-local current streams
static thread_local LeakyStreamInternals** current_streams = nullptr;

// Populates global values and creates a default stream for each device.
// Note: the default stream on each device is signified by a nullptr,
// and so is not created as usual.
// In particular, we don't need to switch devices when creating the
// streams.
// Warning: this function must only be called once!
static void initGlobalStreamState() {
  num_gpus = device_count();
  // Check if the number of GPUs matches the expected compile-time max number
  // of GPUs.
  TORCH_CHECK(
      num_gpus <= C10_COMPILE_TIME_MAX_GPUS,
      "Number of CUDA devices on the machine is larger than the compiled "
      "max number of gpus expected (",
      C10_COMPILE_TIME_MAX_GPUS,
      "). Increase that and recompile.");

  // Initializes default streams
  for (const auto i : c10::irange(num_gpus)) {
    default_streams[i].device_index = i;
    low_priority_counters[i] = 0;
    high_priority_counters[i] = 0;
  }
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
static void initDeviceStreamState(DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  CUDAGuard device_guard{device_index};

  for (const auto i : c10::irange(kStreamsPerPool)) {
    auto& lowpri_stream = low_priority_streams[device_index][i];
    auto& hipri_stream = high_priority_streams[device_index][i];

    lowpri_stream.device_index = device_index;
    hipri_stream.device_index = device_index;

    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &lowpri_stream.stream, kDefaultFlags, kLowPriority));
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &hipri_stream.stream, kDefaultFlags, kHighPriority));
  }
}

// Init front-end to ensure initialization only occurs once
static void initCUDAStreamsOnce() {
  // Inits default streams (once, globally)
  std::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  current_streams =
      (LeakyStreamInternals**)malloc(num_gpus * sizeof(LeakyStreamInternals*));
  for (const auto i : c10::irange(num_gpus)) {
    current_streams[i] = &default_streams[i];
  }
}

// Helper to verify the GPU index is valid
static inline void check_gpu(DeviceIndex device_index) {
  AT_ASSERT(device_index >= 0 && device_index < num_gpus);
}

// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in CUDAStream.h)
static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

// See Note [StreamId assignment]
LeakyStreamInternals* CUDAStream_internals(CUDAStream s) {
  c10::DeviceIndex device_index = s.device_index();
  StreamIdType st = streamIdType(s.unwrap().id());
  size_t si = streamIdIndex(s.unwrap().id());
  switch (st) {
    case StreamIdType::DEFAULT:
      TORCH_INTERNAL_ASSERT(
          si == 0,
          "Unrecognized stream ",
          s.unwrap(),
          " (I think this should be the default stream, but I got a non-zero index ",
          si,
          ").",
          " Did you manufacture the StreamId yourself?  Don't do that; use the",
          " official API like c10::cuda::getStreamFromPool() to get a new stream.");
      return &default_streams[device_index];
    case StreamIdType::LOW:
      return &low_priority_streams[device_index][si];
    case StreamIdType::HIGH:
      return &high_priority_streams[device_index][si];
    default:
      TORCH_INTERNAL_ASSERT(
          0,
          "Unrecognized stream ",
          s.unwrap(),
          " (I didn't recognize the stream type, ",
          st,
          ")");
  }
}

CUDAStream CUDAStream_fromInternals(const LeakyStreamInternals* ptr) {
  return CUDAStream(
      CUDAStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::CUDA, ptr->device_index),
          CUDAStream_getStreamId(ptr)));
}

} // anonymous namespace

cudaStream_t CUDAStream::stream() const {
  auto ptr = CUDAStream_internals(*this);
  AT_ASSERT(ptr);
  return ptr->stream;
}

// Returns a stream from the requested pool
// Note: when called the first time on a device, this will create the
// stream pools for that device.
CUDAStream getStreamFromPool(
    const bool isHighPriority,
    DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1)
    device_index = current_device();
  check_gpu(device_index);

  // Initializes the stream pools (once)
  std::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);

  if (isHighPriority) {
    const auto idx = get_idx(high_priority_counters[device_index]);
    return CUDAStream_fromInternals(&high_priority_streams[device_index][idx]);
  }

  const auto idx = get_idx(low_priority_counters[device_index]);
  return CUDAStream_fromInternals(&low_priority_streams[device_index][idx]);
}

CUDAStream getDefaultCUDAStream(DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_gpu(device_index);
  return CUDAStream_fromInternals(&default_streams[device_index]);
}
CUDAStream getCurrentCUDAStream(DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_gpu(device_index);
  return CUDAStream_fromInternals(current_streams[device_index]);
}

void setCurrentCUDAStream(CUDAStream stream) {
  initCUDAStreamsOnce();
  auto ptr = CUDAStream_internals(stream);
  AT_ASSERT(ptr);
  current_streams[ptr->device_index] = ptr;
}

std::ostream& operator<<(std::ostream& stream, const CUDAStream& s) {
  return stream << s.unwrap();
}

} // namespace cuda
} // namespace c10
