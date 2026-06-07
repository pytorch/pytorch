#include <cstddef>
#include <iostream>
#include <vector>

namespace torch::aot_inductor {

struct AOTIPerThreadEventCache {
  std::vector<cudaEvent_t> events;

  void create(std::size_t slot) {
    AOTI_RUNTIME_CUDA_CHECK(
        cudaEventCreateWithFlags(&events[slot], cudaEventDisableTiming));
  }

  void ensure(std::size_t count) {
    const std::size_t old_size = events.size();
    if (old_size >= count) {
      return;
    }
    events.resize(count, nullptr);
    for (std::size_t i = old_size; i < count; ++i) {
      if (i == 0) {
        continue;
      }
      create(i);
    }
  }

  cudaEvent_t get(int idx) {
    const std::size_t slot = static_cast<std::size_t>(idx);
    ensure(slot + 1);
    if (events[slot] == nullptr) {
      create(slot);
    }
    return events[slot];
  }

  ~AOTIPerThreadEventCache() {
    // Best-effort cleanup at thread exit; destructors must not raise and CUDA
    // may already be tearing down.
    for (auto event : events) {
      if (event == nullptr) {
        continue;
      }
      auto code = cudaEventDestroy(event);
      if (code != cudaSuccess) {
        std::cerr << "Failed to destroy CUDA event in AOTInductor stream cache: "
                  << cudaGetErrorString(code) << '\n';
      }
    }
  }
};

struct AOTIPerThreadStreamCache {
  std::vector<cudaStream_t> streams;

  void ensure(std::size_t count, int /*device_idx*/) {
    // device_idx is unused for now: cudaStream is bound to the current device
    // at creation time, and the caller sets the device via AOTICudaGuard before
    // this runs.
    const std::size_t old_size = streams.size();
    if (old_size >= count) {
      return;
    }
    streams.resize(count, nullptr);
    for (std::size_t i = old_size; i < count; ++i) {
      if (i == 0) {
        continue;
      }
      AOTI_RUNTIME_CUDA_CHECK(
          cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }
  }

  cudaStream_t get(int idx, int device_idx) {
    AOTI_RUNTIME_CHECK(
        idx > 0,
        "AOTI aux stream cache slot 0 is reserved for the caller stream");
    const std::size_t slot = static_cast<std::size_t>(idx);
    ensure(slot + 1, device_idx);
    return streams[slot];
  }

  ~AOTIPerThreadStreamCache() {
    for (auto stream : streams) {
      if (stream == nullptr) {
        continue;
      }
      auto code = cudaStreamDestroy(stream);
      if (code != cudaSuccess) {
        std::cerr
            << "Failed to destroy CUDA stream in AOTInductor stream cache: "
            << cudaGetErrorString(code) << '\n';
      }
    }
  }
};

} // namespace torch::aot_inductor
