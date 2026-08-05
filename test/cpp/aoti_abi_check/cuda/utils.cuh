#pragma once

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <vector>

#define SKIP_IF_NO_CUDA_DEVICE()                                     \
  do {                                                               \
    int device_count = 0;                                            \
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||          \
        device_count == 0) {                                         \
      GTEST_SKIP() << "No CUDA device available";                    \
    }                                                                \
  } while (0)

#define CUDA_EXPECT_OK(expr) EXPECT_EQ((expr), cudaSuccess)

namespace torch::test {

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::vector<T>& host) : size_(host.size()) {
    CUDA_EXPECT_OK(cudaMalloc(&ptr_, size_ * sizeof(T)));
    CUDA_EXPECT_OK(
        cudaMemcpy(ptr_, host.data(), size_ * sizeof(T), cudaMemcpyDefault));
  }
  ~DeviceBuffer() {
    cudaFree(ptr_);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() {
    return ptr_;
  }
  const T* get() const {
    return ptr_;
  }

  std::vector<T> to_host() const {
    std::vector<T> host(size_);
    CUDA_EXPECT_OK(
        cudaMemcpy(host.data(), ptr_, size_ * sizeof(T), cudaMemcpyDefault));
    return host;
  }

 private:
  T* ptr_ = nullptr;
  size_t size_;
};

} // namespace torch::test
