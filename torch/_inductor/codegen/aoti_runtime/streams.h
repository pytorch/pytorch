#ifndef AOTI_RUNTIME_STREAMS_H
#define AOTI_RUNTIME_STREAMS_H

#include <cstddef>
#include <iostream>
#include <vector>

namespace torch::aot_inductor {

struct AOTIPerThreadEventCache {
  std::vector<std::vector<cudaEvent_t>> events_by_device;

  std::vector<cudaEvent_t>& events(int device_idx) {
    AOTI_RUNTIME_CHECK(device_idx >= 0, "AOTI event cache device index < 0");
    const auto slot = static_cast<std::size_t>(device_idx);
    if (events_by_device.size() <= slot) {
      events_by_device.resize(slot + 1);
    }
    return events_by_device[slot];
  }

  void create(std::vector<cudaEvent_t>& events, std::size_t slot) {
    AOTI_RUNTIME_CUDA_CHECK(
        cudaEventCreateWithFlags(&events[slot], cudaEventDisableTiming));
  }

  cudaEvent_t get(int event_idx, int device_idx) {
    AOTI_RUNTIME_CHECK(event_idx >= 0, "AOTI event cache event index < 0");
    auto& device_events = events(device_idx);
    const auto slot = static_cast<std::size_t>(event_idx);
    if (device_events.size() <= slot) {
      device_events.resize(slot + 1, nullptr);
    }
    if (device_events[slot] == nullptr) {
      create(device_events, slot);
    }
    return device_events[slot];
  }

  ~AOTIPerThreadEventCache() {
    for (auto& device_events : events_by_device) {
      for (auto event : device_events) {
        if (event == nullptr) {
          continue;
        }
        auto code = cudaEventDestroy(event);
        if (code != cudaSuccess) {
          std::cerr
              << "Failed to destroy CUDA event in AOTInductor stream cache: "
              << cudaGetErrorString(code) << '\n';
        }
      }
    }
  }
};

struct AOTIPerThreadStreamCache {
  std::vector<std::vector<cudaStream_t>> streams_by_device;

  void check_not_capturing(cudaStream_t caller_stream) {
    cudaStreamCaptureStatus capture_status;
    AOTI_RUNTIME_CUDA_CHECK(
        cudaStreamIsCapturing(caller_stream, &capture_status));
    AOTI_RUNTIME_CHECK(
        capture_status == cudaStreamCaptureStatusNone,
        "AOTI user streams are not supported during CUDA graph capture");
  }

  std::vector<cudaStream_t>& streams(int device_idx) {
    AOTI_RUNTIME_CHECK(device_idx >= 0, "AOTI stream cache device index < 0");
    const auto slot = static_cast<std::size_t>(device_idx);
    if (streams_by_device.size() <= slot) {
      streams_by_device.resize(slot + 1);
    }
    return streams_by_device[slot];
  }

  void ensure(std::size_t count, int device_idx, cudaStream_t caller_stream) {
    check_not_capturing(caller_stream);
    auto& device_streams = streams(device_idx);
    const std::size_t old_size = device_streams.size();
    if (old_size >= count) {
      return;
    }
    device_streams.resize(count, nullptr);
    for (std::size_t i = old_size; i < count; ++i) {
      if (i == 0) {
        continue;
      }
      AOTI_RUNTIME_CUDA_CHECK(
          cudaStreamCreateWithFlags(&device_streams[i], cudaStreamNonBlocking));
    }
  }

  cudaStream_t get(int stream_idx, int device_idx, cudaStream_t caller_stream) {
    AOTI_RUNTIME_CHECK(
        stream_idx > 0,
        "AOTI aux stream cache slot 0 is reserved for the caller stream");
    const auto slot = static_cast<std::size_t>(stream_idx);
    ensure(slot + 1, device_idx, caller_stream);
    return streams(device_idx)[slot];
  }

  ~AOTIPerThreadStreamCache() {
    for (auto& device_streams : streams_by_device) {
      for (auto stream : device_streams) {
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
  }
};

} // namespace torch::aot_inductor

#endif // AOTI_RUNTIME_STREAMS_H
