#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include <iostream>
namespace c10 {
namespace cuda {

namespace {

// Global stream state and constants
static c10::once_flag init_flag;
static DeviceIndex num_gpus = -1;
static constexpr int kStreamsPerPoolBits = 5;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
static constexpr unsigned int kDefaultFlags = cudaStreamNonBlocking;
static constexpr int kStreamTypeBits = 3;

// Note: lower numbers are higher priorities, zero is default priority
static constexpr int kHighPriority = -1;
static constexpr int kLowPriority = 0;

// Non-default streams
// Note: the number of CUDA devices is determined at run time,
// and the low and high priority pools are lazily initialized
// when the first stream is requested for a device.
// The device flags track the initialization of each device, while
// the low and high priority counters track, for each device, the next stream
// in the pool to be returned when a stream is requested (round-robin fashion
// , see the note in CUDAStream.h).
// The streams are "leaked": they are created but never destroyed because the
// destruction of global variables could happen after the CUDA runtime has
// already been destroyed and thus invoking cudaStreamDestroy could lead to a
// crash. It's likely an issue in CUDA, but to be safe - let's just "forget"
// the destruction.
static c10::once_flag device_flags[C10_COMPILE_TIME_MAX_GPUS];
static std::atomic<uint32_t> low_priority_counters[C10_COMPILE_TIME_MAX_GPUS];
static std::atomic<uint32_t> high_priority_counters[C10_COMPILE_TIME_MAX_GPUS];
static cudaStream_t low_priority_streams[C10_COMPILE_TIME_MAX_GPUS]
                                        [kStreamsPerPool];
static cudaStream_t high_priority_streams[C10_COMPILE_TIME_MAX_GPUS]
                                         [kStreamsPerPool];

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 57 bits --  -- 5 bits -----  -- 3 bits --
// zeros          stream id index  StreamIdType
//
// Where StreamIdType:
//  000 = default stream or externally allocated if id[63:3] != 0
//  001 = low priority stream
//  010 = high priority stream
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
//
// Also, external managed stream pointers (cudaStream_t) can be directly stored
// in the Id field so in this case, we need to check the stream alignment.
// The IdType uses an additional bit to match with the 64-bit address alignment
// making easy to identify an external stream when its value (X & 7) > 0
enum class StreamIdType : uint8_t {
  DEFAULT = 0x0,
  LOW = 0x1,
  HIGH = 0x2,
  EXT = 0x3,
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
    case StreamIdType::EXT:
      stream << "EXT";
      break;
    default:
      stream << static_cast<uint8_t>(s);
      break;
  }
  return stream;
}

// StreamId is 64-bit, so we can just rely on regular promotion rules.
// We rely on streamIdIndex and streamIdType being non-negative;
// see Note [Hazard when concatenating signed integers]

static inline StreamIdType streamIdType(StreamId s) {
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  if (s && ((s & mask_for_type) == 0)) {
    // Externally allocated streams have their id being the cudaStream_ptr
    // so the bits corresponding to the type will be 0 and will collide with
    // the default stream.
    return StreamIdType::EXT;
  }
  return static_cast<StreamIdType>(s & mask_for_type);
}

static inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(
      (s >> kStreamTypeBits) & ((1 << kStreamsPerPoolBits) - 1));
}

StreamId makeStreamId(StreamIdType st, size_t si) {
  return (static_cast<StreamId>(si) << kStreamTypeBits) |
      static_cast<StreamId>(st);
}

// Thread-local current streams
static thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

// Populates global values.
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

    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &lowpri_stream, kDefaultFlags, kLowPriority));
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(
        &hipri_stream, kDefaultFlags, kHighPriority));

    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_stream_creation(
          reinterpret_cast<uintptr_t>(lowpri_stream));
      (*interp)->trace_gpu_stream_creation(
          reinterpret_cast<uintptr_t>(hipri_stream));
    }
  }

  low_priority_counters[device_index] = 0;
  high_priority_counters[device_index] = 0;
}

// Init front-end to ensure initialization only occurs once
static void initCUDAStreamsOnce() {
  // Inits default streams (once, globally)
  c10::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  current_streams = std::make_unique<StreamId[]>(num_gpus);
  for (const auto i : c10::irange(num_gpus)) {
    current_streams[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}

// Helper to verify the GPU index is valid
static inline void check_gpu(DeviceIndex device_index) {
  TORCH_INTERNAL_ASSERT(device_index >= 0 && device_index < num_gpus);
}

// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in CUDAStream.h)
static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

CUDAStream CUDAStreamForId(DeviceIndex device_index, StreamId stream_id) {
  return CUDAStream(
      CUDAStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::CUDA, device_index),
          stream_id));
}

} // anonymous namespace

// See Note [StreamId assignment]
cudaStream_t CUDAStream::stream() const {
  c10::DeviceIndex device_index = stream_.device_index();
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  size_t si = streamIdIndex(stream_id);
  switch (st) {
    case StreamIdType::DEFAULT:
      TORCH_INTERNAL_ASSERT(
          si == 0,
          "Unrecognized stream ",
          stream_,
          " (I think this should be the default stream, but I got a non-zero index ",
          si,
          ").",
          " Did you manufacture the StreamId yourself?  Don't do that; use the",
          " official API like c10::cuda::getStreamFromPool() to get a new stream.");
      return nullptr;
    case StreamIdType::LOW:
      return low_priority_streams[device_index][si];
    case StreamIdType::HIGH:
      return high_priority_streams[device_index][si];
    case StreamIdType::EXT:
      return reinterpret_cast<cudaStream_t>(stream_id);
    default:
      TORCH_INTERNAL_ASSERT(
          0,
          "Unrecognized stream ",
          stream_,
          " (I didn't recognize the stream type, ",
          st,
          ")");
  }
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
  c10::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);

  if (isHighPriority) {
    const auto idx = get_idx(high_priority_counters[device_index]);
    return CUDAStreamForId(device_index, makeStreamId(StreamIdType::HIGH, idx));
  }

  const auto idx = get_idx(low_priority_counters[device_index]);
  return CUDAStreamForId(device_index, makeStreamId(StreamIdType::LOW, idx));
}

CUDAStream getStreamFromExternal(
    cudaStream_t ext_stream,
    DeviceIndex device_index) {
  // The stream pointer will be the actual id
  return CUDAStreamForId(device_index, reinterpret_cast<int64_t>(ext_stream));
}

CUDAStream getDefaultCUDAStream(DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_gpu(device_index);
  return CUDAStreamForId(device_index, makeStreamId(StreamIdType::DEFAULT, 0));
}

CUDAStream getCurrentCUDAStream(DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_gpu(device_index);
  return CUDAStreamForId(device_index, current_streams[device_index]);
}

void setCurrentCUDAStream(CUDAStream stream) {
  initCUDAStreamsOnce();
  current_streams[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const CUDAStream& s) {
  return stream << s.unwrap();
}

} // namespace cuda
} // namespace c10
