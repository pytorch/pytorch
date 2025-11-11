#include "OpenRegStream.h"

#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>

namespace c10::openreg {

namespace {

// Global stream state and constants
static c10::once_flag init_flag;

static DeviceIndex num_devices = -1;
static constexpr int kStreamsPerPoolBits = 5;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
static constexpr int kStreamTypeBits = 2;

/*
 * The stream pools are lazily initialized when the first queue is requested
 * for a device. The device flags track the initialization of each device. When
 * a queue is requested, the next queue in the pool to be returned in a
 * round-robin fashion, see Note [Stream Management].
 */
static std::deque<c10::once_flag> device_flags;
static std::vector<std::array<
    std::array<orStream_t, kStreamsPerPool>,
    c10::openreg::max_compile_time_stream_priorities>>
    streams;
static std::deque<
    std::array<std::atomic<uint32_t>, max_compile_time_stream_priorities>>
    priority_counters;

static thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

/*
 * Note [StreamId assignment]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~
 * How do we assign stream IDs?
 *
 * -- 56 bits --    -- 5 bits --     -- 2 bits --     -- 1 bit --
 *     zeros       StreamIdIndex     StreamIdType    Ext/native stream
 *                ignored for ext   ignored for ext
 *
 * Where StreamIdType:
 *  00 = default stream
 *  01 = normal stream
 *  11 = external stream
 *
 * For external stream, StreamID is a orStream_t pointer. This means that last
 * bit will always be 0. So when constructing StreamId for a native stream we
 * set last bit to 1 to distinguish between native and external streams.
 *
 * StreamId is 64-bit, so we can just rely on regular promotion rules.
 * We rely on StreamIdIndex and StreamIdType being non-negative;
 */
using StreamIdIndex = uint8_t;
enum class StreamIdType : uint8_t {
  DEFAULT = 0x0,
  NORMAL = 0x1,
  EXT = 0x3,
};

inline std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  switch (s) {
    case StreamIdType::DEFAULT:
      return stream << "DEFAULT";
    case StreamIdType::NORMAL:
      return stream << "NORMAL";
    case StreamIdType::EXT:
      return stream << "EXT";
    default:
      break;
  }

  return stream << static_cast<int16_t>(s);
}

static inline StreamIdType streamIdType(StreamId s) {
  // Externally allocated streams have their id being the orStream_ptr
  // so the last bit will be 0
  if (!(s & 1)) {
    return StreamIdType(StreamIdType::EXT);
  }

  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto st = static_cast<StreamIdType>((s >> 1) & mask_for_type);
  TORCH_CHECK(
      st == StreamIdType::DEFAULT || st == StreamIdType::NORMAL,
      "invalid StreamId: ",
      s);
  return st;
}

static inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(
      (s >> (kStreamTypeBits + 1)) & ((1 << kStreamsPerPoolBits) - 1));
}

StreamId makeStreamId(StreamIdType st, size_t si) {
  if (st == StreamIdType::EXT) {
    return static_cast<StreamId>(0);
  }

  return (static_cast<StreamId>(si) << (kStreamTypeBits + 1)) |
      (static_cast<StreamId>(st) << 1) | 1;
}

static void initGlobalStreamState() {
  num_devices = device_count();
  device_flags.resize(num_devices);
  streams.resize(num_devices);
  priority_counters.resize(num_devices);
}

static void initSingleDeviceStream(
    int priority,
    DeviceIndex device_index,
    int i) {
  auto& stream = streams[device_index][priority][i];

  OPENREG_CHECK(orStreamCreateWithPriority(&stream, 0, priority));
  priority_counters[device_index][priority] = 0;
}

// Creates stream pools for the specified device. It should be call only once.
static void initDeviceStreamState(DeviceIndex device_index) {
  for (const auto i : c10::irange(kStreamsPerPool)) {
    for (const auto p : c10::irange(max_compile_time_stream_priorities)) {
      initSingleDeviceStream(p, device_index, i);
    }
  }
}

static void initOpenRegStreamsOnce() {
  c10::call_once(init_flag, initGlobalStreamState);

  for (const auto i : c10::irange(num_devices)) {
    c10::call_once(
        device_flags[i], initDeviceStreamState, static_cast<DeviceIndex>(i));
  }

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to the last queue in the "normal
  // priority" queue pool. Note: the queue pool have not been initialized yet.
  // It will be initialized in initDeviceStreamState for the specified device.
  current_streams = std::make_unique<StreamId[]>(num_devices);
  for (const auto i : c10::irange(num_devices)) {
    current_streams[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}

static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

OpenRegStream OpenRegStreamForId(DeviceIndex device_index, StreamId stream_id) {
  return OpenRegStream(
      OpenRegStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::PrivateUse1, device_index),
          stream_id));
}

} // anonymous namespace

// See Note [StreamId assignment]
orStream_t OpenRegStream::stream() const {
  c10::DeviceIndex device_index = stream_.device_index();
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  size_t si = streamIdIndex(stream_id);
  switch (st) {
    // The index 0 stream is default as well.
    case StreamIdType::DEFAULT:
    case StreamIdType::NORMAL:
      return streams[device_index][static_cast<uint8_t>(st)][si];
    case StreamIdType::EXT:
      return reinterpret_cast<orStream_t>(stream_id);
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
// Note: when called the first time on a device, this will create the
// stream pools for that device.
OpenRegStream getStreamFromPool(const int priority, DeviceIndex device_index) {
  initOpenRegStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  auto pri_idx =
      std::clamp(priority, 0, max_compile_time_stream_priorities - 1);
  const auto idx = get_idx(priority_counters[device_index][pri_idx]);
  auto id_type = static_cast<StreamIdType>(pri_idx);
  return OpenRegStreamForId(device_index, makeStreamId(id_type, idx));
}

OpenRegStream getStreamFromPool(const bool isHighPriority, DeviceIndex device) {
  initOpenRegStreamsOnce();
  int priority = 0;
  return getStreamFromPool(priority, device);
}

OpenRegStream getStreamFromExternal(
    orStream_t ext_stream,
    DeviceIndex device_index) {
  return OpenRegStreamForId(
      device_index, reinterpret_cast<int64_t>(ext_stream));
}

OpenRegStream getDefaultOpenRegStream(DeviceIndex device_index) {
  initOpenRegStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  return OpenRegStreamForId(
      device_index, makeStreamId(StreamIdType::DEFAULT, 0));
}

OpenRegStream getCurrentOpenRegStream(DeviceIndex device_index) {
  initOpenRegStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  return OpenRegStreamForId(device_index, current_streams[device_index]);
}

void setCurrentOpenRegStream(OpenRegStream stream) {
  initOpenRegStreamsOnce();
  current_streams[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const OpenRegStream& s) {
  return stream << s.unwrap();
}

} // namespace c10::openreg
