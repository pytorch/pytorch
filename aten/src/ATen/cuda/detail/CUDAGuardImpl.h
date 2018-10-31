#pragma once

#include <c10/detail/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAStream.h>

#include <cuda_runtime_api.h>

namespace at {
namespace cuda {
namespace detail {

struct CUDAGuardImpl final : public c10::detail::DeviceGuardImplInterface {
  CUDAGuardImpl() {}
  DeviceType type() const override {
    return DeviceType::CUDA;
  }
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::CUDA);
    int old_device;
    AT_CUDA_CHECK(cudaGetDevice(&old_device));
    if (old_device != d.index()) {
      AT_CUDA_CHECK(cudaSetDevice(d.index()));
    }
    return Device(DeviceType::CUDA, old_device);
  }
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::CUDA);
    AT_CUDA_CHECK(cudaSetDevice(d.index()));
  }
  void uncheckedSetDevice(Device device) const noexcept override {
    cudaSetDevice(device.index());
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    CUDAStream cs(s);
    // TODO: Don't go through internals if not necessary
    CUDAStream old_stream(CUDAStream_getCurrentStream(s.device().index()));
    CUDAStream_setStream(cs.internals());
    return old_stream.unwrap();
  }
};

}}} // namespace at::cuda::detail
