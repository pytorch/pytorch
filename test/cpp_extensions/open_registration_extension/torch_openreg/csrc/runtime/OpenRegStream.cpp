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
c10::once_flag init_flag;
DeviceIndex num_devices = -1;
constexpr int kStreamsPerPoolBits = 5;
constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
constexpr int kStreamTypeBits = 3;

int max_stream_priorities;

/*
 * The stream pools are lazily initialized when the first queue is requested
 * for a device. The device flags track the initialization of each device. When
 * a queue is requested, the next queue in the pool to be returned in a
 * round-robin fashion, see Note [Stream Management].
 */
std::deque<c10::once_flag> device_flags;
std::vector<std::array<
    std::array<orStream_t, kStreamsPerPool>,
    c10::openreg::max_compile_time_stream_priorities>>
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
 *     zeros       StreamIdIndex     StreamIdType    Ext/Native stream
 *                ignored for ext   ignored for ext
 *
 * StreamIdType:
 *  000 = normal stream
 *  001 = high stream
 *  110 = default stream
 *  111 = external stream

 * The range 000 to 101 is reserved for stream pools of different priorities and can be expanded as needed. (OpenReg currently supports two priorities: 0 and 1)
 *
 * For external stream, StreamID is a orStream_t pointer. This means that last
 * bit will always be 0. So when constructing StreamId for a native stream we
 * set last bit to 1 to distinguish between native and external streams.
 *
 * StreamId is 64-bit, so we can just rely on regular promotion rules.
 * We rely on StreamIdIndex and StreamIdType being non-negative;
 */
using StreamIdIndex = uint8_t;
class StreamIdType {
 private:
  uint8_t stream_type;

 public:
  static const uint8_t DEFAULT = 0x6;
  static const uint8_t EXT = 0x7;

 public:
  StreamIdType(const uint8_t _stream_type) : stream_type(_stream_type) {}

  bool isExt() const {
    return EXT == stream_type;
  }

  bool isDefault() const {
    return DEFAULT == stream_type;
  }

  uint8_t getStreamType() const {
    return stream_type;
  }
};

inline std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  switch (s.getStreamType()) {
    case StreamIdType::DEFAULT:
      return stream << "DEFAULT";
    case StreamIdType::EXT:
      return stream << "EXT";
    default:
      return stream << "PRIORITY" << static_cast<int>(s.getStreamType());
  }
}

inline StreamIdType streamIdType(StreamId s) {
  if (!(s & 1)) {
    return StreamIdType(StreamIdType::EXT);
  }
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto st = (s >> 1) & mask_for_type;
  TORCH_CHECK(
      st == StreamIdType::DEFAULT || (st >= 0 && st < max_stream_priorities),
      "invalid StreamIdType: ",
      st);
  return st;
}

inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(
      (s >> (kStreamTypeBits + 1)) & ((1 << kStreamsPerPoolBits) - 1));
}

StreamId makeStreamId(StreamIdType st, size_t si) {
  return (static_cast<StreamId>(si) << (kStreamTypeBits + 1)) |
      (static_cast<StreamId>(st.getStreamType()) << 1) | 1;
}

void initGlobalStreamState() {
  num_devices = device_count();
  device_flags.resize(num_devices);
  streams.resize(num_devices);
  priority_counters.resize(num_devices);
  int leastPriority = -1, greatestPriority = -1;
  OPENREG_CHECK(
      orDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  auto range = greatestPriority - leastPriority + 1;
  max_stream_priorities = range >= c10::openreg::max_compile_time_stream_priorities
      ? c10::openreg::max_compile_time_stream_priorities
      : range;
}

void initSingleDeviceStream(int priority, DeviceIndex device_index, int i) {
  auto& stream = streams[device_index][priority][i];
  OPENREG_CHECK(orStreamCreateWithPriority(&stream, 0, priority));
  priority_counters[device_index][priority] = 0;
}


// Creates stream pools for the specified device. It should be call only once.
void initDeviceStreamState(DeviceIndex device_index) {
  DeviceGuard device_guard{Device(DeviceType::PrivateUse1, device_index)};
  for (const auto i : c10::irange(kStreamsPerPool)) {
    for (const auto p : c10::irange(max_stream_priorities)) {
      initSingleDeviceStream(p, device_index, i);
    }
  }
}

void initOpenRegStreamsOnce() {
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

inline void check_device(DeviceIndex device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < num_devices,
      "Device index value ",
      static_cast<int>(device_index),
      " is out of index range [0, ",
      static_cast<int>(num_devices),
      ")");
}

uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw = counter++;
  return raw % kStreamsPerPool;
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
  // OpenReg does not support a default stream natively.
  // Here, we designate stream 0 from the priority 0 stream pool to serve as the default stream.
  if(st.isDefault()){
    return streams[device_index][0][0];
  }else if(st.isExt()){
    return reinterpret_cast<orStream_t>(stream_id);
  }else{
    auto streamType = st.getStreamType();
    TORCH_CHECK(
        streamType >= 0 && streamType <= max_stream_priorities,
        "Unrecognized stream ",
        stream_,
        " (I didn't recognize the stream type, ",
        st,
        " with the value ",
        streamType,
        ")");
    return streams[device_index][streamType][si];
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
  auto pri_idx = std::clamp(priority, 0, max_stream_priorities - 1);
  const auto idx = get_idx(priority_counters[device_index][pri_idx]);
  auto id_type = static_cast<StreamIdType>(pri_idx);
  return OpenRegStreamForId(device_index, makeStreamId(id_type, idx));
}

OpenRegStream getStreamFromPool(const bool isHighPriority, DeviceIndex device) {
  initOpenRegStreamsOnce();
  int priority = isHighPriority ? max_stream_priorities - 1 : 0;
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
  check_device(device_index);
  return OpenRegStreamForId(
      device_index, makeStreamId(StreamIdType::DEFAULT, 0));
}

OpenRegStream getCurrentOpenRegStream(DeviceIndex device_index) {
  initOpenRegStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_device(device_index);
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
