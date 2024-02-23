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

// The SYCL queue pools are lazily initialized when the first queue is requested
// for a device. The device flags track the initialization of each device. When
// a queue is requested, the next queue in the pool to be returned in a
// round-robin fashion, see Note [Stream Management].
std::deque<c10::once_flag> device_flags;
std::vector<std::array<
    std::array<std::unique_ptr<sycl::queue>, kStreamsPerPool>,
    max_compile_time_stream_priorities>>
    streams;
std::deque<
    std::array<std::atomic<uint32_t>, max_compile_time_stream_priorities>>
    priority_counters;

thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 57 bits --  -- 5 bits -----  -- 3 bits --
//     zeros      StreamIdIndex    StreamIdType
//
// Where StreamIdType:
//  000 = normal priority queue
//  001 = high priority queue
//
// StreamId is 64-bit, so we can just rely on regular promotion rules.
// We rely on StreamIdIndex and StreamIdType being non-negative;

using StreamIdIndex = uint8_t;
enum class StreamIdType : uint8_t {
  // The higher the type number, the higher the priority.
  NORMAL = 0x0,
  HIGH = 0X1,
};

inline std::ostream& operator<<(std::ostream& stream, StreamIdType q) {
  switch (q) {
    case StreamIdType::NORMAL:
      return stream << "NORMAL";
    case StreamIdType::HIGH:
      return stream << "HIGH";
    default:
      break;
  }
  return stream << static_cast<int16_t>(q);
}

inline StreamIdType streamIdType(StreamId s) {
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto st = static_cast<StreamIdType>(s & mask_for_type);
  TORCH_CHECK(
      st == StreamIdType::NORMAL || st == StreamIdType::HIGH,
      "invalid StreamId: ",
      s);
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
  streams.resize(num_gpus);
  priority_counters.resize(num_gpus);
}

// Creates the reserved SYCL queue pools for the specified device. It should be
// call only once.
void initDeviceStreamState(DeviceIndex device) {
  using namespace sycl::ext::oneapi::property;
  // Need to align with StreamIdType.
  const std::vector<sycl::property_list> properties = {
      {sycl::property::queue::in_order(), queue::priority_normal()},
      {sycl::property::queue::in_order(), queue::priority_high()}};
  for (const auto p : c10::irange(max_compile_time_stream_priorities)) {
    for (const auto i : c10::irange(kStreamsPerPool)) {
      streams[device][p][i] = std::make_unique<sycl::queue>(sycl::queue(
          c10::xpu::get_device_context(),
          c10::xpu::get_raw_device(device),
          c10::xpu::asyncHandler,
          properties[p]));
    }
    priority_counters[device][p] = 0;
  }
}

void initXPUStreamsOnce() {
  c10::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to the last queue in the "normal
  // priority" queue pool. Note: the queue pool have not been initialized yet.
  // It will be initialized in initDeviceStreamState for the specified device.
  current_streams = std::make_unique<StreamId[]>(num_gpus);
  for (const auto i : c10::irange(num_gpus)) {
    // Assigning the current stream to the last one in the pool can be
    // beneficial in certain scenarios, particularly when users initialize their
    // workload to perform computations with the current stream (the last one)
    // and utilize stream (the first one) from the pool for communication, it
    // allows for different streams to overlap in computation and communication.
    current_streams[i] =
        makeStreamId(StreamIdType::NORMAL, kStreamsPerPool - 1);
  }
}

// Creates the reserved sycl queue pools for the specified device to ensure
// initialization only occurs once.
inline void initDeviceStreamOnce(DeviceIndex device) {
  c10::call_once(device_flags[device], initDeviceStreamState, device);
}

inline void check_device(DeviceIndex device) {
  TORCH_CHECK(
      device >= 0 && device < num_gpus,
      "device is out of range, device is ",
      static_cast<int16_t>(device),
      ", total number of device is ",
      static_cast<int16_t>(num_gpus),
      ".");
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

int XPUStream::priority() const {
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  // StreamIdType and priority number are inversely related.
  return -static_cast<int>(st);
}

// See Note [StreamId assignment]
sycl::queue& XPUStream::queue() const {
  DeviceIndex device_index = stream_.device_index();
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  StreamIdIndex si = streamIdIndex(stream_id);
  switch (st) {
    case StreamIdType::NORMAL:
    case StreamIdType::HIGH:
      return *streams[device_index][static_cast<uint8_t>(st)][si];
    default:
      TORCH_CHECK(
          false,
          "Unrecognized stream ",
          stream_,
          " (I didn't recognize the stream type, ",
          st,
          ").",
          " Did you manufacture the StreamId yourself?  Don't do that;");
  }
}

// Returns a stream from the requested pool
// Note: The stream pools will be initialized if needed, at the first invocation
// to this function.
XPUStream getStreamFromPool(const int priority, DeviceIndex device) {
  initXPUStreamsOnce();
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  check_device(device);
  TORCH_CHECK(
      priority <= 0,
      "Expected XPU stream priority to be less than or equal to 0, got ",
      priority);
  // Initializes the stream pools (once)
  initDeviceStreamOnce(device);
  auto priority_idx =
      std::min(-priority, max_compile_time_stream_priorities - 1);
  const auto idx = get_idx(priority_counters[device][priority_idx]);
  auto id_type = static_cast<StreamIdType>(priority_idx);
  return XPUStreamForId(device, makeStreamId(id_type, idx));
}

XPUStream getStreamFromPool(const bool isHighPriority, DeviceIndex device) {
  initXPUStreamsOnce();
  // If isHighPriority is true, return the stream with the highest priority.
  int priority = isHighPriority ? -max_compile_time_stream_priorities + 1 : 0;
  return getStreamFromPool(priority, device);
}

// Note: The stream pools will be initialized if needed, at the first invocation
// to this function.
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

// Note: The stream pools will be initialized if needed, at the first invocation
// to this function.
void setCurrentXPUStream(XPUStream stream) {
  initXPUStreamsOnce();
  current_streams[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const XPUStream& s) {
  return stream << s.unwrap();
}

/*
 * Note [Synchronize Streams on Device]
 *
 * There are two stream pools per device to manage our reserved SYCL queues.
 * When syncStreamsOnDevice is called, all reserved SYCL queues in the pools of
 * the specified device will be blocked, and wait for their synchronizations. We
 * realize the semantics via a loop through the stream pools of the specified
 * device and make each command queue synchronization sequentially.
 *
 * There is a semantic gap with device synchronization because only the SYCL
 * queues we have reserved (in our pools) will be synchronized, rather than
 * synchronizing all SYCL queues on the specified device.
 */

// Note: The stream pools will be initialized if needed, at the first invocation
// to this function.
void syncStreamsOnDevice(DeviceIndex device) {
  initXPUStreamsOnce();
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  check_device(device);
  // Initializes the stream pools (once)
  initDeviceStreamOnce(device);

  // For each device, we have kStreamsPerPool (32) reserved queues per priority.
  for (const auto p : c10::irange(max_compile_time_stream_priorities)) {
    for (const auto i : c10::irange(kStreamsPerPool)) {
      streams[device][p][i]->wait();
    }
  }
}

} // namespace c10::xpu
