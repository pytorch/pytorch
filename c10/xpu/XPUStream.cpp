#include <c10/util/CallOnce.h>
#include <c10/util/irange.h>
#include <c10/xpu/XPUException.h>
#include <c10/xpu/XPUStream.h>

#include <atomic>
#include <deque>
#include <mutex>
#include <vector>

namespace c10::xpu {
namespace {

// Global stream state and constants
c10::once_flag init_flag;
DeviceIndex num_gpus = -1;
constexpr int kStreamsPerPoolBits = 5;
constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
constexpr int kStreamTypeBits = 3;

// The SYCL queue pool is lazily initialized when the first queue is requested
// for a device. The device flags track the initialization of each device. When
// a queue is requested, the next queue in the pool to be returned in a
// round-robin fashion, see Note [Stream Management].
std::deque<c10::once_flag> device_flags;
std::vector<std::array<std::unique_ptr<sycl::queue>, kStreamsPerPool>>
    reserved_streams;
std::deque<std::atomic<uint32_t>> reserved_counters;

thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 57 bits --  -- 5 bits -----  -- 3 bits --
//     zeros      StreamIdIndex    StreamIdType
//
// Where StreamIdType:
//  000 = UNUSED
//  001 = reserved queue
//
// StreamId is 64-bit, so we can just rely on regular promotion rules.
// We rely on StreamIdIndex and StreamIdType being non-negative;

using StreamIdIndex = uint8_t;
enum class StreamIdType : uint8_t {
  UNUSED = 0x0,
  RESERVED = 0x1,
};

inline std::ostream& operator<<(std::ostream& stream, StreamIdType q) {
  switch (q) {
    case StreamIdType::UNUSED:
      stream << "UNUSED";
      break;
    case StreamIdType::RESERVED:
      stream << "RESERVED";
      break;
    default:
      stream << static_cast<uint8_t>(q);
      break;
  }
  return stream;
}

inline StreamIdType streamIdType(StreamId s) {
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto st = static_cast<StreamIdType>(s & mask_for_type);
  TORCH_INTERNAL_ASSERT(st == StreamIdType::RESERVED, "invalid StreamId", s);
  return st;
}

inline StreamIdIndex streamIdIndex(StreamId s) {
  return static_cast<StreamIdIndex>(
      (s >> kStreamTypeBits) & ((1 << kStreamsPerPoolBits) - 1));
}

inline StreamId makeStreamId(StreamIdType st, StreamIdIndex si) {
  return (static_cast<StreamId>(si) << kStreamTypeBits) |
      static_cast<StreamId>(st);
}

void initGlobalStreamState() {
  num_gpus = c10::xpu::device_count();
  device_flags.resize(num_gpus);
  reserved_streams.resize(num_gpus);
  reserved_counters.resize(num_gpus);
}

// Creates the reserved SYCL queue pool for the specified device. It should be
// call only once.
void initDeviceStreamState(DeviceIndex device) {
  // Switches to the requested device so streams are properly associated
  // with it.
  for (const auto i : c10::irange(kStreamsPerPool)) {
    reserved_streams[device][i] = std::make_unique<sycl::queue>(sycl::queue(
        c10::xpu::get_device_context(),
        c10::xpu::get_raw_device(device),
        c10::xpu::asyncHandler,
        {sycl::property::queue::in_order()}));
  }
  reserved_counters[device] = 0;
}

void initXPUStreamsOnce() {
  c10::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to the first queue in the queue pool.
  // Note: the queue pool have not been initialized yet. It will be initialized
  // in initDeviceStreamState for the specified device.
  current_streams = std::make_unique<StreamId[]>(num_gpus);
  for (const auto i : c10::irange(num_gpus)) {
    current_streams[i] = makeStreamId(StreamIdType::RESERVED, 0);
  }
}

// Creates the reserved sycl queue pool for the specified device to ensure
// initialization only occurs once.
inline void initDeviceStreamOnce(DeviceIndex device) {
  c10::call_once(device_flags[device], initDeviceStreamState, device);
}

inline void check_device(DeviceIndex device) {
  TORCH_INTERNAL_ASSERT(device >= 0 && device < num_gpus);
}

uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

XPUStream XPUStreamForId(DeviceIndex device_index, StreamId stream_id) {
  return XPUStream(
      XPUStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::XPU, device_index),
          stream_id));
}

} // anonymous namespace

// See Note [StreamId assignment]
sycl::queue& XPUStream::queue() const {
  DeviceIndex device_index = stream_.device_index();
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  StreamIdIndex si = streamIdIndex(stream_id);
  switch (st) {
    case StreamIdType::UNUSED:
      TORCH_INTERNAL_ASSERT(
          0,
          "Unrecognized stream ",
          stream_,
          " (I didn't recognize the stream type, ",
          st,
          ").",
          " Did you manufacture the StreamId yourself?  Don't do that;");
    case StreamIdType::RESERVED:
      return *reserved_streams[device_index][si];
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
// Note: when called the first time on a device, this will create the stream
// pool for that device.
XPUStream getStreamFromPool(const bool isHighPriority, DeviceIndex device) {
  initXPUStreamsOnce();
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  check_device(device);
  TORCH_CHECK(
      !isHighPriority,
      "Currently, high priority stream is not supported in XPU backend.");

  // Initializes the stream pool (once)
  initDeviceStreamOnce(device);
  const auto idx = get_idx(reserved_counters[device]);
  return XPUStreamForId(device, makeStreamId(StreamIdType::RESERVED, idx));
}

// Note: when called the first time on a device, this will create the stream
// pool for that device.
XPUStream getCurrentXPUStream(DeviceIndex device) {
  initXPUStreamsOnce();
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  check_device(device);
  // Initializes the stream pool (once)
  initDeviceStreamOnce(device);
  return XPUStreamForId(device, current_streams[device]);
}

void setCurrentXPUStream(XPUStream stream) {
  initXPUStreamsOnce();
  current_streams[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const XPUStream& s) {
  return stream << s.unwrap();
}

// Note: when called the first time on a device, this will create the stream
// pool for that device.
void device_synchronize(DeviceIndex device) {
  initXPUStreamsOnce();
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  check_device(device);
  // Initializes the stream pool (once)
  initDeviceStreamOnce(device);

  // For each device, we have kStreamsPerPool (32) reserved queues.
  for (const auto i : c10::irange(kStreamsPerPool)) {
    reserved_streams[device][i]->wait();
  }
}

} // namespace c10::xpu
