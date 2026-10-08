// Host state for the side-aware elementwise schedule: the runtime toggle, the
// registry of side-striped ranges, the per-device SM side map, and the
// per-(device, stream) claim queues.
#include <ATen/native/cuda/SideAware.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace at::native::side_aware {

namespace {

constexpr int kSamples = 9;

std::atomic<bool>& enabled_flag() {
  static std::atomic<bool> flag{[] {
    const char* env = std::getenv("PYTORCH_SIDE_AWARE");
    return env == nullptr || std::strcmp(env, "0") != 0;
  }()};
  return flag;
}

std::atomic<uint64_t> launches{0};

// One slot per registered range, guarded by a sequence lock: writers (under
// Ranges::mutex) make `seq` odd, update the slot, and make it even again;
// readers retry until they see the same even `seq` before and after reading.
// end == 0 marks a free slot.
constexpr int kMaxRanges = 8;

struct RangeSlot {
  std::atomic<uint32_t> seq{0};
  std::atomic<uintptr_t> base{0}, end{0};
};

struct Ranges {
  std::mutex mutex;
  std::array<RangeSlot, kMaxRanges> slots;
  std::atomic<uintptr_t> probe{0}; // base of the first registered range
};

Ranges& ranges(c10::DeviceIndex device) {
  static std::array<Ranges, C10_COMPILE_TIME_MAX_GPUS> instance;
  return instance[device];
}

void write_slot(RangeSlot& slot, uintptr_t base, uintptr_t end) {
  const uint32_t seq = slot.seq.load(std::memory_order_relaxed);
  slot.seq.store(seq + 1, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  slot.base.store(base, std::memory_order_relaxed);
  slot.end.store(end, std::memory_order_relaxed);
  slot.seq.store(seq + 2, std::memory_order_release);
}

void check_device(c10::DeviceIndex device) {
  TORCH_CHECK(device >= 0 && device < C10_COMPILE_TIME_MAX_GPUS, "invalid device ", int(device));
}

struct DeviceState {
  std::once_flag init;
  bool valid = false;
  int sm_count = 0;
  signed char* sm_side = nullptr; // device, kMaxSms entries
  std::vector<int> host_sides;
  std::mutex mutex;
  std::unordered_map<c10::StreamId, unsigned long long*> queues; // keyed by c10 stream id
};

DeviceState& state(c10::DeviceIndex device) {
  static std::array<DeviceState, C10_COMPILE_TIME_MAX_GPUS> states;
  return states[device];
}

// Cold-load latency in cycles. The second clock read is predicated on the
// loaded value, so it waits for the load.
__device__ unsigned load_latency(const unsigned* addr) {
  unsigned value;
  unsigned long long start, stop = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(addr) : "memory");
  asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 1; @p mov.u64 %0, %%clock64; }"
               : "+l"(stop) : "r"(value) : "memory");
  return unsigned(stop - start);
}

// The first CTA on each SM times kSamples cold loads to each probe page (side 0
// and side 1, interleaved) and assigns the SM to the side with the lower median.
// Each SM reads its own lines, so no load hits a line another SM brought into L2.
__global__ void probe_sm_sides(const char* page0, const char* page1, unsigned* claimed, int* sm_side) {
  const unsigned sm = smid();
  if (threadIdx.x != 0 || sm >= kMaxSms || atomicCAS(&claimed[sm], 0u, 1u) != 0) {
    return;
  }
  unsigned samples[2][kSamples];
  for (int i = 0; i < kSamples; ++i) {
    const size_t offset = (size_t(sm) * kSamples + i) * 896 % kPageBytes; // 7 lines apart
    samples[0][i] = load_latency(reinterpret_cast<const unsigned*>(page0 + offset));
    samples[1][i] = load_latency(reinterpret_cast<const unsigned*>(page1 + offset));
  }
  for (int s = 0; s < 2; ++s) { // insertion sort
    for (int i = 1; i < kSamples; ++i) {
      for (int j = i; j > 0 && samples[s][j] < samples[s][j - 1]; --j) {
        const unsigned tmp = samples[s][j];
        samples[s][j] = samples[s][j - 1];
        samples[s][j - 1] = tmp;
      }
    }
  }
  sm_side[sm] = samples[1][kSamples / 2] < samples[0][kSamples / 2] ? 1 : 0;
}

// Builds the SM side map of `device` from the probe pages of its first
// registered range.
void init_device(c10::DeviceIndex device, DeviceState& st) {
  c10::cuda::CUDAGuard guard(device);
  const char* page0 = reinterpret_cast<const char*>(ranges(device).probe.load(std::memory_order_acquire));
  const cudaDeviceProp* prop = at::cuda::getDeviceProperties(device);
  // The kernel body is compiled for sm_107 only.
  if (page0 == nullptr || prop->major != 10 || prop->minor != 7 ||
      prop->multiProcessorCount > kMaxSms) {
    return;
  }
  const int sm_count = prop->multiProcessorCount;
  // Raw cudaMalloc: these buffers must not come from the caller's MemPool.
  const size_t scratch_bytes = size_t(prop->l2CacheSize) * 8;
  void* scratch = nullptr;
  int* sides = nullptr;
  unsigned* claimed = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&scratch, scratch_bytes));
  C10_CUDA_CHECK(cudaMalloc(&sides, kMaxSms * sizeof(int)));
  C10_CUDA_CHECK(cudaMalloc(&claimed, kMaxSms * sizeof(unsigned)));
  C10_CUDA_CHECK(cudaMemset(sides, 0xff, kMaxSms * sizeof(int))); // -1: not probed
  C10_CUDA_CHECK(cudaMemset(claimed, 0, kMaxSms * sizeof(unsigned)));
  C10_CUDA_CHECK(cudaMemset((void*)page0, 0x5a, 2 * kPageBytes)); // never the predicate's 1
  C10_CUDA_CHECK(cudaMemset(scratch, 1, scratch_bytes)); // 8x L2: the probe loads miss L2
  probe_sm_sides<<<sm_count * 16, 32>>>(page0, page0 + kPageBytes, claimed, sides);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  std::vector<int> host(kMaxSms);
  C10_CUDA_CHECK(cudaMemcpy(host.data(), sides, kMaxSms * sizeof(int), cudaMemcpyDeviceToHost));
  C10_CUDA_CHECK(cudaFree(scratch));
  C10_CUDA_CHECK(cudaFree(sides));
  C10_CUDA_CHECK(cudaFree(claimed));
  host.resize(sm_count);
  const auto side1 = std::count(host.begin(), host.end(), 1);
  const auto unprobed = std::count(host.begin(), host.end(), -1);
  if (unprobed != 0 || side1 < sm_count / 4 || side1 > sm_count * 3 / 4) {
    TORCH_WARN(
        "side-aware schedule disabled on device ", device, ": SM side map has ",
        unprobed, " unprobed SMs and ", side1, "/", sm_count, " SMs on side 1");
    return;
  }
  signed char table[kMaxSms] = {};
  for (int sm = 0; sm < sm_count; ++sm) {
    table[sm] = static_cast<signed char>(host[sm]);
  }
  C10_CUDA_CHECK(cudaMalloc(&st.sm_side, kMaxSms));
  C10_CUDA_CHECK(cudaMemcpy(st.sm_side, table, kMaxSms, cudaMemcpyHostToDevice));
  st.host_sides = std::move(host);
  st.sm_count = sm_count;
  st.valid = true;
}

} // namespace

void register_striped_range(c10::DeviceIndex device, uintptr_t base, size_t size) {
  check_device(device);
  TORCH_CHECK(
      base != 0 && base % kBlockAlign == 0 && size >= kBlockAlign && size % kBlockAlign == 0 &&
          base + size > base,
      "side-striped range must be 4 MiB aligned and a nonzero multiple of 4 MiB");
  Ranges& r = ranges(device);
  std::lock_guard<std::mutex> lock(r.mutex);
  RangeSlot* free_slot = nullptr;
  for (auto& slot : r.slots) {
    const uintptr_t b = slot.base.load(std::memory_order_relaxed);
    const uintptr_t e = slot.end.load(std::memory_order_relaxed);
    TORCH_CHECK(e == 0 || base + size <= b || base >= e, "side-striped range overlaps a registered one");
    if (e == 0 && free_slot == nullptr) {
      free_slot = &slot;
    }
  }
  TORCH_CHECK(free_slot != nullptr, "at most ", kMaxRanges, " side-striped ranges per device");
  write_slot(*free_slot, base, base + size);
  uintptr_t no_probe = 0;
  r.probe.compare_exchange_strong(no_probe, base);
}

void unregister_striped_range(c10::DeviceIndex device, uintptr_t base) {
  check_device(device);
  Ranges& r = ranges(device);
  std::lock_guard<std::mutex> lock(r.mutex);
  for (auto& slot : r.slots) {
    if (slot.end.load(std::memory_order_relaxed) != 0 && slot.base.load(std::memory_order_relaxed) == base) {
      write_slot(slot, 0, 0);
      if (r.probe.load(std::memory_order_relaxed) == base) { // probe another live range, if any
        uintptr_t next = 0;
        for (const auto& other : r.slots) {
          if (next == 0 && other.end.load(std::memory_order_relaxed) != 0) {
            next = other.base.load(std::memory_order_relaxed);
          }
        }
        r.probe.store(next, std::memory_order_release);
      }
      return;
    }
  }
  TORCH_CHECK(false, "no side-striped range registered at this base");
}

bool contains(c10::DeviceIndex device, const void* ptr, size_t bytes) {
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  for (const auto& slot : ranges(device).slots) {
    uintptr_t base = 0, end = 0;
    uint32_t seq = 0;
    do {
      seq = slot.seq.load(std::memory_order_acquire);
      base = slot.base.load(std::memory_order_relaxed);
      end = slot.end.load(std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_acquire);
    } while ((seq & 1) != 0 || slot.seq.load(std::memory_order_relaxed) != seq);
    if (end != 0 && addr >= base && addr < end && bytes <= end - addr) {
      return true;
    }
  }
  return false;
}

bool enabled() {
  return enabled_flag().load(std::memory_order_relaxed);
}

void set_enabled(bool value) {
  enabled_flag().store(value);
}

uint64_t launch_count() {
  return launches.load();
}

void count_launch() {
  launches.fetch_add(1, std::memory_order_relaxed);
}

Calibration::Calibration() {
  static const bool forced = [] {
    const char* env = std::getenv("PYTORCH_SIDE_AWARE_CALIBRATE");
    return env != nullptr && std::strcmp(env, "0") == 0;
  }();
  if (forced) {
    decision_.store(1, std::memory_order_relaxed);
  }
}

// Under mutex_. On a bailout, handed-out stop events may still be recorded,
// so the events are leaked rather than destroyed.
bool Calibration::decide(bool side) {
  decision_.store(side ? 1 : -1, std::memory_order_release);
  return side;
}

bool Calibration::use_side(
    c10::DeviceIndex device, cudaStream_t stream, size_t bytes, cudaEvent_t* stop, const char* name) {
  *stop = nullptr;
  if (const int decision = decision_.load(std::memory_order_acquire); decision != 0) {
    return decision > 0;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (const int decision = decision_.load(std::memory_order_relaxed); decision != 0) {
    return decision > 0;
  }
  if (issued_ == 0) {
    device_ = device;
  } else if (device != device_) {
    return decide(false); // events of two devices cannot be compared
  }
  constexpr int kTimed = 2 * kSamplesPerPath;
  if (issued_ < 2 * kTimed) {
    // Launches go D D* S S* D D* S S*: only the second of each pair is timed
    // (*), so a timed kernel follows one of its own kind as in steady state.
    // Timing strictly alternating launches biases both kernels low (each one
    // pays for the other's cache and clock state) and can flip near-ties.
    const int launch = issued_++;
    const bool side = (launch / 2) % 2 == 1;
    if (launch % 2 == 1) {
      const int i = launch / 2; // sample; i % 2 is the path
      C10_CUDA_CHECK(cudaEventCreate(&events_[i][0]));
      C10_CUDA_CHECK(cudaEventCreate(&events_[i][1]));
      C10_CUDA_CHECK(cudaEventRecord(events_[i][0], stream));
      bytes_[i] = bytes;
      *stop = events_[i][1];
    }
    return side;
  }
  // Every sample is issued; until all stops are recorded and have run, keep
  // the default kernel, and give up after kMaxPendingLaunches.
  if (++pending_ > kMaxPendingLaunches) {
    return decide(false);
  }
  if (recorded_.load(std::memory_order_acquire) < kTimed) {
    return false;
  }
  double best[2] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  for (int i = 0; i < kTimed; ++i) {
    float ms = 0;
    const cudaError_t err = cudaEventElapsedTime(&ms, events_[i][0], events_[i][1]);
    if (err != cudaSuccess) {
      // As in at::cuda::CUDAEvent::query: clear only the error this call raised.
      (void)cudaGetLastError();
      if (err == cudaErrorNotReady) {
        return false;
      }
      TORCH_WARN("side-aware calibration failed (", cudaGetErrorString(err), "); using the default kernel");
      return decide(false);
    }
    best[i % 2] = std::min(best[i % 2], double(ms) / double(bytes_[i])); // [0] default, [1] side
  }
  for (auto& pair : events_) {
    C10_CUDA_CHECK(cudaEventDestroy(pair[0]));
    C10_CUDA_CHECK(cudaEventDestroy(pair[1]));
  }
  const bool side = best[1] < best[0];
  if (std::getenv("PYTORCH_SIDE_AWARE_DEBUG") != nullptr) {
    const char* with = std::strstr(name, "[with F = ");
    std::fprintf(stderr, "[side_aware] calibrated: default %.2f, side %.2f TB/s -> %s: %.240s\n",
                 1e-9 / best[0], 1e-9 / best[1], side ? "side" : "default", with ? with : name);
  }
  return decide(side);
}

bool get_context(const c10::cuda::CUDAStream& stream, Context* ctx) {
  const c10::DeviceIndex device = stream.device_index();
  // Lazily allocated queues are not graph-safe; keep captures on the default path.
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture));
  if (capture != cudaStreamCaptureStatusNone) {
    return false;
  }
  DeviceState& st = state(device);
  // First-use cudaMalloc/cudaMemset must not invalidate another thread's capture.
  c10::cuda::CUDAStreamCaptureModeGuard relaxed(cudaStreamCaptureModeRelaxed);
  std::call_once(st.init, [&] {
    try {
      init_device(device, st);
    } catch (const c10::Error& e) { // fall back to the default kernel rather than fail the op
      TORCH_WARN("side-aware schedule disabled on device ", device, ": ", e.what_without_backtrace());
    }
  });
  if (!st.valid) {
    return false;
  }
  std::lock_guard<std::mutex> lock(st.mutex);
  unsigned long long*& queues = st.queues[stream.id()];
  if (queues == nullptr) {
    C10_CUDA_CHECK(cudaMalloc(&queues, 3 * sizeof(unsigned long long)));
    // Ordered before the first kernel on this stream; launches leave it zeroed.
    C10_CUDA_CHECK(cudaMemsetAsync(queues, 0, 3 * sizeof(unsigned long long), stream.stream()));
  }
  *ctx = Context{st.sm_side, queues, st.sm_count};
  return true;
}

std::vector<int> sm_sides(c10::DeviceIndex device) {
  return state(device).host_sides;
}

} // namespace at::native::side_aware
