#pragma once

// Host-side state of the side-aware elementwise schedule (see SideAware.cuh).

#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Export.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace at::native::side_aware {

constexpr size_t kPageBytes = size_t(2) << 20; // one locality-domain page
constexpr size_t kBlockAlign = 2 * kPageBytes; // a side-0 page followed by a side-1 page

// Registry of side-striped VA ranges: ranges whose 2 MiB page at address
// `addr` is backed by locality domain (addr >> 21) & 1. An allocator
// (torch.cuda.memory.LocalityInterleavedAllocator) registers each range it
// reserves; operands inside a registered range are eligible for the
// side-aware schedule. `base` must be 4 MiB aligned, `size` a multiple of
// 4 MiB, and the first 4 MiB (page 0 on side 0, page 1 on side 1) mapped and
// reserved for the allocator's own use: the first live range registered on a
// device serves as the probe for that device's SM side map, which is built
// once, on the first eligible launch. Only the probe pages and pages holding
// tensors need to be mapped. Registration takes a mutex; contains() is
// lock-free and is called on every eligible launch.
TORCH_CUDA_CPP_API void register_striped_range(c10::DeviceIndex device, uintptr_t base, size_t size);
TORCH_CUDA_CPP_API void unregister_striped_range(c10::DeviceIndex device, uintptr_t base);
// True if [ptr, ptr + bytes) lies inside one registered range of `device`.
TORCH_CUDA_CPP_API bool contains(c10::DeviceIndex device, const void* ptr, size_t bytes);

// Per-(device, stream) launch state.
struct Context {
  const signed char* sm_side; // device array: SM id -> side
  unsigned long long* queues; // [0], [1]: next chunk per side; [2]: CTAs done. Zero between launches.
  int sm_count;
};

TORCH_CUDA_CPP_API bool enabled();
TORCH_CUDA_CPP_API void set_enabled(bool value);
// Side-aware kernel launches (for tests).
TORCH_CUDA_CPP_API uint64_t launch_count();
TORCH_CUDA_CPP_API void count_launch();

// Per-functor choice between the side-aware kernel and the default kernel.
// The side kernel cannot help when the functor's arithmetic, not memory,
// bounds the default kernel, and can lose when its registers spill; neither
// is visible at compile time. So the first 4 * kSamplesPerPath eligible
// launches of each side_aware_kernel instantiation time both paths with CUDA
// events (see use_side); once those events have completed, the faster path
// per byte is used for good. Calibration falls back to the default path for
// good if its samples span two devices or are still pending after
// kMaxPendingLaunches further launches. Both paths give bitwise identical
// results. PYTORCH_SIDE_AWARE_CALIBRATE=0 skips this (always side).
class TORCH_CUDA_CPP_API Calibration {
 public:
  Calibration();
  // Path for one eligible launch on `stream` of `device`. For a timed sample,
  // records a start event and sets *stop, which the caller records right
  // after launching the chosen kernel and then reports with stop_recorded().
  bool use_side(c10::DeviceIndex device, cudaStream_t stream, size_t bytes, cudaEvent_t* stop, const char* name);
  void stop_recorded() {
    recorded_.fetch_add(1, std::memory_order_release);
  }

 private:
  bool decide(bool side);
  static constexpr int kSamplesPerPath = 2;
  static constexpr int kMaxPendingLaunches = 32;
  std::mutex mutex_;
  std::atomic<int> decision_{0}; // 0 undecided, 1 side, -1 default
  std::atomic<int> recorded_{0}; // stop events recorded
  int issued_ = 0; // calibration launches so far
  int pending_ = 0; // launches since the last sample was issued
  c10::DeviceIndex device_ = -1; // device of the samples
  cudaEvent_t events_[2 * kSamplesPerPath][2] = {};
  size_t bytes_[2 * kSamplesPerPath] = {};
};

// Declared before a possible fallback to the default kernel: records `stop`
// once that launch has been enqueued (end of launch_vectorized_kernel).
struct DefaultLaunchTimer {
  cudaEvent_t stop = nullptr;
  cudaStream_t stream = nullptr;
  Calibration* calibration = nullptr;
  ~DefaultLaunchTimer() {
    if (stop != nullptr && cudaEventRecord(stop, stream) == cudaSuccess) {
      calibration->stop_recorded();
    }
  }
};
// Fills `ctx` for `stream`. False if its device is not an sm_107 device with
// a registered range and an SM side map, or the stream is capturing a graph.
TORCH_CUDA_CPP_API bool get_context(
    const c10::cuda::CUDAStream& stream,
    Context* ctx);
// Host copy of the SM side map of `device` (empty if none).
TORCH_CUDA_CPP_API std::vector<int> sm_sides(c10::DeviceIndex device);

} // namespace at::native::side_aware
