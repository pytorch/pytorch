// Host state for the side-aware elementwise schedule: the runtime toggle, the
// per-device SM side map, and the per-(device, stream) claim queues.
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
    const char* env = std::getenv("PYTORCH_CUDA_SIDE_AWARE");
    return env == nullptr || std::strcmp(env, "0") != 0;
  }()};
  return flag;
}

std::atomic<uint64_t> launches{0};

bool calibrate_enabled() {
  static const bool value = [] {
    const char* env = std::getenv("PYTORCH_SIDE_AWARE_CALIBRATE");
    return env == nullptr || std::strcmp(env, "0") != 0;
  }();
  return value;
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

// Builds the SM side map of `device` from its arena's probe pages.
void init_device(c10::DeviceIndex device, DeviceState& st) {
  c10::cuda::CUDAGuard guard(device);
  const char* page0 = static_cast<const char*>(c10::cuda::LocalityAllocator::arena_base(device));
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

bool Calibration::use_side(cudaStream_t stream, size_t bytes, cudaEvent_t* stop, const char* name) {
  *stop = nullptr;
  if (const int decision = decision_.load(std::memory_order_acquire); decision != 0) {
    return decision > 0;
  }
  if (!calibrate_enabled()) {
    return true;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (const int decision = decision_.load(std::memory_order_relaxed); decision != 0) {
    return decision > 0;
  }
  if (issued_ < 4 * kSamplesPerPath) {
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
  // Every sample is issued; until all have run (or another thread has not
  // recorded its stop event yet), keep the default kernel.
  double best[2] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  for (int i = 0; i < 2 * kSamplesPerPath; ++i) {
    float ms = 0;
    const cudaError_t err = cudaEventElapsedTime(&ms, events_[i][0], events_[i][1]);
    if (err != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    best[i % 2] = std::min(best[i % 2], double(ms) / double(bytes_[i])); // [0] default, [1] side
  }
  for (auto& pair : events_) {
    C10_CUDA_CHECK(cudaEventDestroy(pair[0]));
    C10_CUDA_CHECK(cudaEventDestroy(pair[1]));
  }
  const bool side = best[1] < best[0];
  decision_.store(side ? 1 : -1, std::memory_order_release);
  if (std::getenv("PYTORCH_SIDE_AWARE_DEBUG") != nullptr) {
    const char* with = std::strstr(name, "[with F = ");
    std::fprintf(stderr, "[side_aware] calibrated: default %.2f, side %.2f TB/s -> %s: %.240s\n",
                 1e-9 / best[0], 1e-9 / best[1], side ? "side" : "default", with ? with : name);
  }
  return side;
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
  std::call_once(st.init, [&] { init_device(device, st); });
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
