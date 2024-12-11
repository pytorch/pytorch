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

/*
 * Note [StreamId assignment]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~
 * How do we assign stream IDs?
 *
 * -- 55 bits --    -- 5 bits --     -- 3 bits --     -- 1 bit --
 *     zeros       StreamIdIndex     StreamIdType    Ext/native stream
 *                ignored for ext   ignored for ext
 *
 * Where StreamIdType:
 *  000 = low priority queue
 *  001 = normal priority queue
 *  010 = high priority queue
 *  111 = external queue
 *
 * For external stream, StreamID is a sycl::queue* pointer. This means that last
 * bit will always be 0. So when constructing StreamId for a native stream we
 * set last bit to 1 to distinguish between native and external streams. For
 * more details, see Note [External XPU Stream].
 *
 * StreamId is 64-bit, so we can just rely on regular promotion rules.
 * We rely on StreamIdIndex and StreamIdType being non-negative;
 */

/*
 * Note [XPU Stream priorities]
 * XPU stream priority levels are defined based on the following design
 * principles:
 *   1. Higher priority number indicates lower priority.
 *   2. The default priority, `normal`, corresponds to a priority number of 0.
 *   3. StreamIdType and priority number are inversely related.
 *
 * This relationship can be summarized as follows:
 * -- priority type --    -- priority number --    -- type number --
 *        low                     1                       0
 *       normal                   0                       1
 *        high                   -1                       2
 */

using StreamIdIndex = uint8_t;
enum class StreamIdType : uint8_t {
  // The higher the type number, the higher the priority for the native stream.
  LOW = 0x0,
  NORMAL = 0x1,
  HIGH = 0x2,
  // For an external stream, the last bit of StreamId is 0, whose priority is
  // queried at runtime.
  EXT = 0x7,
};

inline std::ostream& operator<<(std::ostream& stream, StreamIdType q) {
  switch (q) {
    case StreamIdType::LOW:
      return stream << "LOW";
    case StreamIdType::NORMAL:
      return stream << "NORMAL";
    case StreamIdType::HIGH:
      return stream << "HIGH";
    case StreamIdType::EXT:
      return stream << "EXT";
    default:
      break;
  }
  return stream << static_cast<int16_t>(q);
}

inline StreamIdType streamIdType(StreamId s) {
  // Externally allocated streams have their id being the sycl:queue* pointer.
  // So the last bit will be 0.
  if ((!(s & 1) && s)) {
    return StreamIdType(StreamIdType::EXT);
  }
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto st = static_cast<StreamIdType>((s >> 1) & mask_for_type);
  TORCH_CHECK(
      st == StreamIdType::NORMAL || st == StreamIdType::HIGH ||
          st == StreamIdType::LOW,
      "invalid StreamId: ",
      s);
  return st;
}

inline StreamIdIndex streamIdIndex(StreamId s) {
  return static_cast<StreamIdIndex>(
      (s >> (kStreamTypeBits + 1)) & ((1 << kStreamsPerPoolBits) - 1));
}

inline StreamId makeStreamId(StreamIdType st, StreamIdIndex si) {
  return (static_cast<StreamId>(si) << (kStreamTypeBits + 1)) |
      (static_cast<StreamId>(st) << 1) | 1;
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
      {sycl::property::queue::in_order(), queue::priority_low()},
      {sycl::property::queue::in_order(), queue::priority_normal()},
      {sycl::property::queue::in_order(), queue::priority_high()}};
  TORCH_CHECK(
      properties.size() == max_compile_time_stream_priorities,
      "The number of stream priorities should be equal to max_compile_time_stream_priorities");
  for (const auto p : c10::irange(max_compile_time_stream_priorities)) {
    for (const auto i : c10::irange(kStreamsPerPool)) {
      auto& stream = streams[device][p][i];
      stream = std::make_unique<sycl::queue>(sycl::queue(
          c10::xpu::get_device_context(),
          c10::xpu::get_raw_device(device),
          c10::xpu::asyncHandler,
          properties[p]));
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_stream_creation(
            c10::kXPU, reinterpret_cast<uintptr_t>(stream.get()));
      }
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
  if (C10_UNLIKELY(st == StreamIdType::EXT)) {
    // Query external stream priority
    using namespace sycl::ext::oneapi::property;
    if (queue().has_property<queue::priority_normal>()) {
      st = StreamIdType::NORMAL;
    } else if (queue().has_property<queue::priority_high>()) {
      st = StreamIdType::HIGH;
    } else if (queue().has_property<queue::priority_low>()) {
      st = StreamIdType::LOW;
    } else {
      // Default priority for SYCL queue is normal.
      st = StreamIdType::NORMAL;
    }
  }
  // See Note [XPU Stream priorities]
  return -static_cast<int>(st) + 1;
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
    // See Note [External XPU Stream]
    case StreamIdType::EXT:
      return *(reinterpret_cast<sycl::queue*>(stream_id));
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
  check_device_index(device);
  // Initializes the stream pools (once)
  initDeviceStreamOnce(device);
  // See Note [XPU Stream priorities]
  auto priority_idx = std::max(
      std::min(-priority + 1, max_compile_time_stream_priorities - 1), 0);
  const auto idx = get_idx(priority_counters[device][priority_idx]);
  auto id_type = static_cast<StreamIdType>(priority_idx);
  return XPUStreamForId(device, makeStreamId(id_type, idx));
}

XPUStream getStreamFromPool(const bool isHighPriority, DeviceIndex device) {
  initXPUStreamsOnce();
  // If isHighPriority is true, return the stream with the highest priority.
  // See Note [XPU Stream priorities]
  int priority = isHighPriority ? -max_compile_time_stream_priorities + 2 : 0;
  return getStreamFromPool(priority, device);
}

/*
 * Note [External XPU Stream]
 *
 * An external XPUStream is a wrapper around an external SYCL queue that was not
 * created by PyTorch. This design enables interoperability with other libraries
 * by allowing PyTorch to work seamlessly with SYCL queues created outside of
 * its control.
 *
 * Key design requirements include:
 *   1. Allowing retrieval of the its SYCL queue from the external XPUStream.
 *   2. Supporting conversion between an external XPUStream and a `c10::Stream`.
 *   3. Ensuring compatibility with the `get/setCurrentXPUStream` methods.
 *   4. Enabling memory caching allocation through the external XPUStream.
 *
 * To address requirements (1) and (2), we associate the external SYCL queue
 * pointer with the `stream_id`. It is the user's responsibility to ensure that
 * the referenced SYCL queue remains alive while the corresponding XPUStream, or
 * any c10::Stream derived from it, is in use.
 *
 * However, this approach introduces the following limitations:
 *
 *   1. Different SYCL queue pointers will result in distinct XPUStream
 * instances, even if the SYCL queues they dereference are equivalent.
 *   2. Memory blocks allocated by one external XPUStream CANNOT be reused by
 * other non-equivalent XPUStreams, even if they originate from the same SYCL
 * queue object.
 */

XPUStream getStreamFromExternal(
    sycl::queue* ext_queue,
    DeviceIndex device_index) {
  // The sycl::queue* will be the actual id

  TORCH_CHECK(ext_queue, "External sycl::queue* must not be a nullptr.");
  TORCH_CHECK(
      ext_queue->is_in_order(), "External SYCL queue must be in-order.");
  TORCH_CHECK(
      ext_queue->get_context() == c10::xpu::get_device_context(),
      "External SYCL queue must be created with the same context as the PyTorch XPU used.");
  TORCH_CHECK(
      ext_queue->get_device() == c10::xpu::get_raw_device(device_index),
      "External SYCL queue doesn't match the given device index.");
  return XPUStreamForId(device_index, reinterpret_cast<int64_t>(ext_queue));
}

// Note: The stream pools will be initialized if needed, at the first invocation
// to this function.
XPUStream getCurrentXPUStream(DeviceIndex device) {
  initXPUStreamsOnce();
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  check_device_index(device);
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
  check_device_index(device);
  // Initializes the stream pools (once)
  initDeviceStreamOnce(device);

  // For each device, we have kStreamsPerPool (32) reserved queues per priority.
  for (const auto p : c10::irange(max_compile_time_stream_priorities)) {
    for (const auto i : c10::irange(kStreamsPerPool)) {
      streams[device][p][i]->wait();
    }
  }
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_device_synchronization(c10::kXPU);
  }
}

} // namespace c10::xpu
