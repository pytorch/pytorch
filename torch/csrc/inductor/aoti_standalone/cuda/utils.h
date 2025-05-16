#pragma once
#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include <c10/core/Device.h>

namespace torch::standalone {

inline void throw_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(cudaGetErrorString(err)));
  }
}

// Used in destructor because can't throw in destructor
inline void print_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
    // NOLINTNEXTLINE(performance-avoid-endl)
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
}

class AOTICudaGuard {
 public:
  AOTICudaGuard(int32_t device_index) : device_(device_index) {
    // Save original device
    throw_cuda_error(cudaGetDevice(&prev_device_));
    // Switch to the target device if necessary
    if (prev_device_ != device_) {
      throw_cuda_error(cudaSetDevice(device_));
    }
  }

  AOTICudaGuard(const c10::Device& device)
      : AOTICudaGuard(static_cast<int32_t>(device.index())) {}

  AOTICudaGuard() = delete;
  AOTICudaGuard(const AOTICudaGuard&) = delete;
  AOTICudaGuard& operator=(const AOTICudaGuard&) = delete;
  AOTICudaGuard(AOTICudaGuard&& other) = delete;
  AOTICudaGuard& operator=(AOTICudaGuard&& other) = delete;

  ~AOTICudaGuard() {
    // Restore the original device if necessary
    if (prev_device_ != device_) {
      print_cuda_error(cudaSetDevice(prev_device_));
    }
  }

  void set_index(int32_t device_index) {
    device_ = device_index;
    throw_cuda_error(cudaSetDevice(device_));
  }

 private:
  int32_t prev_device_ = 0;
  int32_t device_;
};

inline std::unordered_map<int, cudaStream_t> current_streams;

// Get the current stream for a specific device
inline cudaStream_t get_current_stream(int32_t device) {
  auto it = current_streams.find(device);
  return (it != current_streams.end()) ? it->second : 0; // Default stream is 0
}

// Set the current stream for a specific device
inline void set_current_stream(int32_t device, cudaStream_t stream) {
  current_streams[device] = stream;
}

class AOTICudaStreamGuard {
 public:
  AOTICudaStreamGuard(cudaStream_t stream, int32_t device_index = 0)
      : stream_(stream), device_(device_index) {
    // Save original device
    throw_cuda_error(cudaGetDevice(&prev_device_));

    // Switch to the target device if necessary
    if (prev_device_ != device_) {
      throw_cuda_error(cudaSetDevice(device_));
    }

    // Save the original stream for the current device
    prev_stream_ = get_current_stream(device_);
    // Set the new stream
    set_current_stream(device_, stream_);
  }

  ~AOTICudaStreamGuard() {
    // Restore the original stream for the current device
    set_current_stream(device_, prev_stream_);

    // Restore the original device if necessary
    if (prev_device_ != device_) {
      print_cuda_error(cudaSetDevice(prev_device_));
    }
  }

  AOTICudaStreamGuard() = delete;
  AOTICudaStreamGuard(const AOTICudaStreamGuard&) = delete;
  AOTICudaStreamGuard& operator=(const AOTICudaStreamGuard&) = delete;
  AOTICudaStreamGuard(AOTICudaStreamGuard&& other) = delete;
  AOTICudaStreamGuard& operator=(AOTICudaStreamGuard&& other) = delete;

 private:
  cudaStream_t prev_stream_; // Original stream on the target device
  cudaStream_t stream_; // Target stream to set
  int32_t prev_device_ = 0; // Original device at construction
  int32_t device_; // Target device for the guard
};

class AOTICudaStream {
 public:
  AOTICudaStream(int32_t device_index = 0) : device_index_(device_index) {
    throw_cuda_error(cudaSetDevice(device_index_));
    throw_cuda_error(cudaStreamCreate(&stream_));
  }

  ~AOTICudaStream() {
    if (stream_) {
      print_cuda_error(cudaStreamDestroy(stream_));
    }
  }

  // Disable copy constructor and copy assignment
  AOTICudaStream(const AOTICudaStream&) = delete;
  AOTICudaStream& operator=(const AOTICudaStream&) = delete;

  // Move constructor
  AOTICudaStream(AOTICudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
  }

  // Move assignment
  AOTICudaStream& operator=(AOTICudaStream&& other) noexcept {
    if (this != &other) {
      if (stream_) {
        print_cuda_error(cudaStreamDestroy(stream_));
      }
      stream_ = other.stream_;
      other.stream_ = nullptr;
    }
    return *this;
  }

  cudaStream_t get() const {
    return stream_;
  }

 private:
  int32_t device_index_{0};
  cudaStream_t stream_{nullptr};
};

#define CUDA_CHECK(EXPR)                                            \
  do {                                                              \
    const cudaError_t __err = EXPR;                                 \
    if (__err != cudaSuccess) {                                     \
      throw std::runtime_error(                                     \
          "CUDA error: " + std::string(cudaGetErrorString(__err))); \
    }                                                               \
  } while (0)

} // namespace torch::standalone
#endif // USE_CUDA
