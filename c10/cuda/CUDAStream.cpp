#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <array>
#include <atomic>
#include <cstdint>

namespace c10::cuda {

namespace {

// Global stream state and constants
DeviceIndex num_gpus = -1;
constexpr int kStreamsPerPoolBits = 5;
constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
constexpr unsigned int kDefaultFlags = cudaStreamNonBlocking;
constexpr int kStreamTypeBits = 4;

int max_stream_priorities;
// Runtime-resolved per-priority round-robin wrap points. Equal to
// kStreamsPerPool on CUDA. On ROCm the default-priority pool (index 0, HIP
// normal) is GPU_MAX_HW_QUEUES-1 because the null stream permanently holds one
// of that priority's hw queues, while higher-priority pools are
// GPU_MAX_HW_QUEUES. See Note [HIP Stream Pool]. Filled by
// initGlobalStreamState before any use.
static_assert(
    c10::cuda::max_compile_time_stream_priorities == 4,
    "update streams_per_pool initializer");
std::array<int, c10::cuda::max_compile_time_stream_priorities>
    streams_per_pool =
        {kStreamsPerPool, kStreamsPerPool, kStreamsPerPool, kStreamsPerPool};

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
// Tracks per-device one-time eager init of the stream pools.
std::array<c10::once_flag, C10_COMPILE_TIME_MAX_GPUS> device_flags;
std::array<
    std::array<std::atomic<uint32_t>, C10_COMPILE_TIME_MAX_GPUS>,
    c10::cuda::max_compile_time_stream_priorities>
    priority_counters;

// Per priority, per device: bitmask of reserved pool slots (bit i set => slot i
// is reserved and excluded from the round-robin in get_idx). Reservations are
// taken by reserveStreamFromPool and ref-counted down by releaseReservedStream;
// the underlying stream is never destroyed, so a stale bit only affects which
// slot round-robin hands out, never correctness. See Note [HIP Stream Pool].
std::array<
    std::array<std::atomic<uint32_t>, C10_COMPILE_TIME_MAX_GPUS>,
    c10::cuda::max_compile_time_stream_priorities>
    reserved_slots;

std::array<
    std::array<
        std::array<cudaStream_t, kStreamsPerPool>,
        C10_COMPILE_TIME_MAX_GPUS>,
    c10::cuda::max_compile_time_stream_priorities>
    streams;

// Note [HIP Stream Pool]
// ~~~~~~~~~~~~~~~~~~~~~~~
// HIP keeps a pool of at most GPU_MAX_HW_QUEUES (default 4) hsa_queue_t per
// stream priority. A new hipStream gets its own queue while that priority's
// pool is below the cap; once the cap is hit, further streams SHARE an existing
// queue (the HIP runtime picks one by least ref-count, breaking ties by raw
// queue pointer, which differs per process -> non-deterministic across ranks).
// Two streams on one queue serialize. So to get both determinism and
// concurrency we keep each priority's pooled-stream count within its
// private-queue budget, so the runtime never enters the sharing/tie-break path:
// every pooled stream gets its own queue via the deterministic grow path.
//
// The budget differs by priority because the null/default stream is an ordinary
// HIP-normal-priority stream that is created first and lives for the whole
// process, permanently holding one of the normal pool's queues:
//  - default priority (index 0 -> HIP normal): GPU_MAX_HW_QUEUES - 1
//  - higher priorities (HIP high): GPU_MAX_HW_QUEUES
// (Verified on ROCm 6.4/7.0/7.2; clr 'develop' makes the null queue explicitly
// dedicated but keeps the same budget since it does not raise
// GPU_MAX_HW_QUEUES.)
//
// The size defaults to the HIP runtime's GPU_MAX_HW_QUEUES default (4) and can
// be overridden via GPU_MAX_HW_QUEUES or PYTORCH_HIP_STREAMS_PER_POOL, clamped
// to kStreamsPerPool (the StreamId index field is only kStreamsPerPoolBits
// wide). Each priority's streams are created eagerly in index order at first
// use, so the per-rank stream->queue structure is reproducible.

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 54 bits --  -- 5 bits -----  -- 4 bits --     --1 bit --
// zeros          stream id index  StreamIdType     Ext/native stream
//                ignored for ext   ignored for ext
// for external stream, StreamID is a cudaStream_t pointer
// this means that last bit will always be 0
// so when constructing StreamId for a native stream we set last bit to 1
// to distinguish between native and external streams
//
//
// We are obligated to treat the stream ID 0 as the default stream, per the
// invariant specified in c10::Stream, so this is one exception to
// "last bit = 1 for native streams". However, all other numbers are entirely
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

class StreamIdType {
  // StreamIdType encodes whether this stream is DEFAULT, EXTernal or
  // for all other native streams, the stream priority (higher value is higher
  // priority)
 private:
  uint8_t stream_type;

 public:
  static const uint8_t DEFAULT = 0x0;
  static const uint8_t EXT = 0xF;

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

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  if (s.isDefault()) {
    stream << "DEFAULT";
  } else if (s.isExt()) {
    stream << "EXT";
  } else {
    stream << "PRIORITY " << static_cast<int>(s.getStreamType());
  }
  return stream;
}

// StreamId is 64-bit, so we can just rely on regular promotion rules.
// We rely on streamIdIndex and streamIdType being non-negative;
// see Note [Hazard when concatenating signed integers]

inline StreamIdType streamIdType(StreamId s) {
  // Externally allocated streams have their id being the cudaStream_ptr
  // so the last bit will be 0
  if ((!(s & 1)) && s) {
    return StreamIdType(StreamIdType::EXT);
  }
  // last bit is external/internal stream, the mask should start from second
  // rightmost bit
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto val = (s >> 1) & mask_for_type;
  TORCH_CHECK(val || !(s & 1), "invalid StreamId", s);
  return StreamIdType(val);
}

inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(
      (s >> (kStreamTypeBits + 1)) & ((1 << kStreamsPerPoolBits) - 1));
}

StreamId makeStreamId(StreamIdType st, size_t si) {
  if (st.isDefault()) {
    return static_cast<StreamId>(0);
  }
  return (static_cast<StreamId>(si) << (kStreamTypeBits + 1)) |
      static_cast<StreamId>(st.getStreamType() << 1) | 1;
}

// Thread-local current streams
// NOLINTNEXTLINE(*-arrays)
thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

// Populates global values.
// Warning: this function must only be called once!
void initGlobalStreamState() {
  num_gpus = device_count();
  // Check if the number of GPUs matches the expected compile-time max number
  // of GPUs.
  TORCH_CHECK(
      num_gpus <= C10_COMPILE_TIME_MAX_GPUS,
      "Number of CUDA devices on the machine is larger than the compiled "
      "max number of gpus expected (",
      C10_COMPILE_TIME_MAX_GPUS,
      "). Increase that and recompile.");
  int leastPriority = -1, greatestPriority = -1;
  C10_CUDA_CHECK(
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  // Note [HIP stream priorities]
  // HIP stream priorities are 1=low, 0=default, -1=high which differs from CUDA
  // which is 0=default, -1=high, -2=higher etc.
  // Clamp leastPriority to 0 for HIP.
#ifdef USE_ROCM
  leastPriority = 0;
#endif
  // greatestPriority is negative
  auto range = leastPriority - greatestPriority + 1;
  max_stream_priorities = range >= c10::cuda::max_compile_time_stream_priorities
      ? c10::cuda::max_compile_time_stream_priorities
      : range;

  // See Note [HIP Stream Pool]
#ifdef USE_ROCM
  // Each per-priority pool is sized to the number of hsa_queues the HIP runtime
  // backs for that priority, so every pooled stream gets a private queue (no
  // runtime sharing / tie-break). Defaults to the HIP runtime's own default of
  // 4 when neither env var is set; GPU_MAX_HW_QUEUES (the env the runtime
  // itself reads) or the PYTORCH_HIP_STREAMS_PER_POOL override changes the
  // count.
  int q = 4; // HIP runtime default for GPU_MAX_HW_QUEUES
  auto env = c10::utils::get_env("PYTORCH_HIP_STREAMS_PER_POOL");
  if (!env.has_value()) {
    env = c10::utils::get_env("GPU_MAX_HW_QUEUES");
  }
  if (env.has_value()) {
    try {
      q = std::stoi(env.value());
    } catch (const std::exception&) {
      TORCH_WARN(
          "Ignoring invalid HIP streams-per-pool value '", env.value(), "'.");
    }
  }
  if (q > kStreamsPerPool) {
    TORCH_WARN_ONCE(
        "Requested ",
        q,
        " HIP streams per pool exceeds the maximum of ",
        kStreamsPerPool,
        "; clamping.");
  }
  q = std::clamp(q, 1, kStreamsPerPool);
  for (const auto p :
       c10::irange(c10::cuda::max_compile_time_stream_priorities)) {
    // The default-priority pool (index 0 -> HIP normal) shares its hw-queue
    // pool with the null stream, which permanently holds one queue; higher
    // priorities have no null stream.
    streams_per_pool[p] = (p == 0) ? std::max(1, q - 1) : q;
  }
#endif
}

// Init a single CUDA or HIP stream
// See Note [HIP Stream Pool]
void initSingleStream(int p, DeviceIndex device_index, int i) {
  CUDAGuard device_guard(device_index);
  auto& stream = streams[p][device_index][i];
  auto pri = -p; // lower number is higher priority

  C10_CUDA_CHECK(cudaStreamCreateWithPriority(&stream, kDefaultFlags, pri));
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_stream_creation(
        c10::kCUDA, reinterpret_cast<uintptr_t>(stream));
    priority_counters[p][device_index] = 0;
  }
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
void initDeviceStreamState(DeviceIndex device_index) {
  for (const auto p : c10::irange(max_stream_priorities)) {
    for (const auto i : c10::irange(streams_per_pool[p])) {
      initSingleStream(p, device_index, i);
    }
  }
}

// Init front-end to ensure initialization only occurs once
void initCUDAStreamsOnce() {
  // Inits default streams (once, globally)
  auto static init_flag [[maybe_unused]] = [] {
    initGlobalStreamState();
    return true;
  }();

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  // NOLINTNEXTLINE(*-arrays)
  current_streams = std::make_unique<StreamId[]>(num_gpus);
  for (const auto i : c10::irange(num_gpus)) {
    current_streams[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}

// Helper to verify the GPU index is valid
inline void check_gpu(DeviceIndex device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < num_gpus,
      "Device index value ",
      static_cast<int>(device_index),
      " is out of index range [0, ",
      static_cast<int>(num_gpus),
      ")");
}

// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in CUDAStream.h)
uint32_t get_idx(
    std::atomic<uint32_t>& counter,
    int pri_idx,
    DeviceIndex device_index) {
  auto raw_idx = counter++;
  const int pool = streams_per_pool[pri_idx];
  const uint32_t reserved =
      reserved_slots[pri_idx][device_index].load(std::memory_order_relaxed);
  // Round-robin over the slots not currently reserved; if every slot is
  // reserved fall back to the full pool (degrade to sharing rather than fail).
  int free_count = 0;
  for (const auto i : c10::irange(pool)) {
    if (!((reserved >> i) & 1u)) {
      free_count++;
    }
  }
  if (free_count == 0) {
    return raw_idx % pool;
  }
  uint32_t target = raw_idx % static_cast<uint32_t>(free_count);
  uint32_t seen = 0;
  for (const auto i : c10::irange(pool)) {
    if (!((reserved >> i) & 1u)) {
      if (seen == target) {
        return static_cast<uint32_t>(i);
      }
      seen++;
    }
  }
  return raw_idx % pool; // unreachable
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

bool CUDAStream::query() const {
  DeviceGuard guard{stream_.device()};
  cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaStreamQuery(stream()));

  if (err == cudaSuccess) {
    return true;
  } else if (err != cudaErrorNotReady) {
    C10_CUDA_CHECK(err);
  } else {
    // ignore and clear the error if not ready
    (void)cudaGetLastError();
  }

  return false;
}

void CUDAStream::synchronize() const {
  DeviceGuard guard{stream_.device()};
  c10::cuda::stream_synchronize(stream());
}

// See Note [StreamId assignment]
cudaStream_t CUDAStream::stream() const {
  c10::DeviceIndex device_index = stream_.device_index();
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  size_t si = streamIdIndex(stream_id);
  initCUDAStreamsOnce();
  check_gpu(device_index);
  if (st.isDefault()) {
    TORCH_CHECK(
        si == 0,
        "Unrecognized stream ",
        stream_,
        " (I think this should be the default stream, but I got a non-zero index ",
        si,
        ").",
        " Did you manufacture the StreamId yourself?  Don't do that; use the",
        " official API like c10::cuda::getStreamFromPool() to get a new stream.");
    return nullptr;
  } else if (st.isExt()) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<cudaStream_t>(stream_id);
  } else {
    auto streamType = st.getStreamType();
    TORCH_CHECK(
        streamType >= 1 && streamType <= max_stream_priorities,
        "Unrecognized stream ",
        stream_,
        " (I didn't recognize the stream type, ",
        st,
        " with the value ",
        streamType,
        ")");
#ifdef USE_ROCM
    // See Note [HIP Stream Pool]: ensure the device's pool is created
    // (idempotent); covers pooled StreamIds reconstructed via unpack3 without a
    // prior getStreamFromPool on this device.
    c10::call_once(
        device_flags[device_index], initDeviceStreamState, device_index);
#endif
    return streams[st.getStreamType() - 1][device_index][si];
  }
}

// Returns a stream from the requested pool
// Note: when called the first time on a device, this will create the
// stream pools for that device.
CUDAStream getStreamFromPool(const int priority, DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    c10::cuda::SetTargetDevice();
  }
  check_gpu(device_index);
  // See Note [HIP Stream Pool]: eagerly create the pool (capped per priority on
  // ROCm) once per device, so the stream->queue assignment is fixed up front.
  c10::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);
  auto pri_idx = std::clamp(-priority, 0, max_stream_priorities - 1);
  const auto idx =
      get_idx(priority_counters[pri_idx][device_index], pri_idx, device_index);
  StreamIdType id_type = StreamIdType(pri_idx + 1);
  return CUDAStreamForId(device_index, makeStreamId(id_type, idx));
}

CUDAStream getStreamFromPool(const bool isHighPriority, DeviceIndex device) {
  initCUDAStreamsOnce();
  int priority = isHighPriority ? -max_stream_priorities + 1 : 0;
  return getStreamFromPool(priority, device);
}

int getStreamsPerPool(int priority) {
  initCUDAStreamsOnce();
  auto pri_idx = std::clamp(-priority, 0, max_stream_priorities - 1);
  return streams_per_pool[pri_idx];
}

// Reserve a stream from the requested priority pool. The returned stream is
// excluded from the round-robin getStreamFromPool until releaseReservedStream
// is called for it, so reserving distinct streams (or reserving against
// transient ones) guarantees they do not share a slot. On ROCm each slot is a
// distinct hsa_queue, so reserved streams run concurrently. See Note [HIP
// Stream Pool].
CUDAStream reserveStreamFromPool(const int priority, DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    c10::cuda::SetTargetDevice();
  }
  check_gpu(device_index);
  // See Note [HIP Stream Pool]: reservations index into the eagerly created
  // pool.
  c10::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);
  auto pri_idx = std::clamp(-priority, 0, max_stream_priorities - 1);
  const int pool = streams_per_pool[pri_idx];
  auto& slots = reserved_slots[pri_idx][device_index];
  uint32_t cur = slots.load(std::memory_order_relaxed);
  while (true) {
    int slot = -1;
    for (const auto i : c10::irange(pool)) {
      if (!((cur >> i) & 1u)) {
        slot = i;
        break;
      }
    }
    if (slot < 0) {
#ifdef USE_ROCM
      TORCH_CHECK(
          false,
          "Cannot reserve another priority ",
          priority,
          " stream; all ",
          pool,
          " HIP queues for that priority are already reserved. Increase "
          "GPU_MAX_HW_QUEUES to at least ",
          (pri_idx == 0 ? pool + 2 : pool + 1),
          ".");
#else
      TORCH_CHECK(
          false,
          "Cannot reserve another priority ",
          priority,
          " stream; all ",
          pool,
          " pool streams for that priority are already reserved.");
#endif
    }
    uint32_t desired = cur | (1u << slot);
    if (slots.compare_exchange_weak(cur, desired, std::memory_order_acq_rel)) {
      StreamIdType id_type = StreamIdType(pri_idx + 1);
      return CUDAStreamForId(device_index, makeStreamId(id_type, slot));
    }
    // compare_exchange_weak reloaded `cur`; retry with the fresh value.
  }
}

CUDAStream reserveStreamFromPool(
    const bool isHighPriority,
    DeviceIndex device) {
  initCUDAStreamsOnce();
  int priority = isHighPriority ? -max_stream_priorities + 1 : 0;
  return reserveStreamFromPool(priority, device);
}

// Drop a reservation taken by reserveStreamFromPool, returning the slot to the
// round-robin pool. Idempotent for non-pool streams (default/external) and safe
// to over-call: it only clears a bit, never destroys the underlying stream.
void releaseReservedStream(CUDAStream stream) {
  StreamId id = stream.id();
  StreamIdType st = streamIdType(id);
  if (st.isDefault() || st.isExt()) {
    return;
  }
  int pri_idx = st.getStreamType() - 1;
  auto device_index = stream.device_index();
  if (pri_idx < 0 || pri_idx >= max_stream_priorities || device_index < 0 ||
      device_index >= num_gpus) {
    return;
  }
  size_t si = streamIdIndex(id);
  reserved_slots[pri_idx][device_index].fetch_and(
      ~(1u << si), std::memory_order_acq_rel);
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
    c10::cuda::SetTargetDevice();
  }
  check_gpu(device_index);
  return CUDAStreamForId(device_index, makeStreamId(StreamIdType::DEFAULT, 0));
}

CUDAStream getCurrentCUDAStream(DeviceIndex device_index) {
  initCUDAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    c10::cuda::SetTargetDevice();
  }
  check_gpu(device_index);
  return CUDAStreamForId(device_index, current_streams[device_index]);
}

void setCurrentCUDAStream(CUDAStream stream) {
  initCUDAStreamsOnce();
  auto device_index = stream.device_index();
  check_gpu(device_index);
  current_streams[device_index] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const CUDAStream& s) {
  return stream << s.unwrap();
}

} // namespace c10::cuda
