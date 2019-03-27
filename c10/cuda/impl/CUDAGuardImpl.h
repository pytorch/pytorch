#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime_api.h>

namespace c10 {
namespace cuda {
namespace impl {

struct CUDAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::CUDA;
  CUDAGuardImpl() {}
  CUDAGuardImpl(DeviceType t) {
    AT_ASSERT(t == DeviceType::CUDA);
  }
  DeviceType type() const override {
    return DeviceType::CUDA;
  }
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::CUDA);
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      C10_CUDA_CHECK(cudaSetDevice(d.index()));
    }
    return old_device;
  }
  Device getDevice() const override {
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    return Device(DeviceType::CUDA, device);
  }
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::CUDA);
    C10_CUDA_CHECK(cudaSetDevice(d.index()));
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    cudaError_t __err = cudaSetDevice(d.index());
    if (__err != cudaSuccess) {
      AT_WARN("CUDA error: ", cudaGetErrorString(__err));
    }
  }
  Stream getStream(Device d) const noexcept override {
    return getCurrentCUDAStream().unwrap();
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    CUDAStream cs(s);
    auto old_stream = getCurrentCUDAStream(s.device().index());
    setCurrentCUDAStream(cs);
    return old_stream.unwrap();
  }
  DeviceIndex deviceCount() const override {
    int deviceCnt;
    C10_CUDA_CHECK(cudaGetDeviceCount(&deviceCnt));
    return deviceCnt;
  }
};

}}} // namespace c10::cuda::impl
