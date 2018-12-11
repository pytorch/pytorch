#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.hpp>

#include <ATen/cuda/CUDAConfig.h>

#if AT_ROCM_ENABLED()
#include <c10/hip/impl/HIPGuardImpl.h>
#endif

namespace at { namespace cuda {

/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

cudaDeviceProp* getCurrentDeviceProperties() {
  return THCState_getCurrentDeviceProperties(at::globalContext().getTHCState());
}

cudaDeviceProp* getDeviceProperties(int64_t device) {
  return THCState_getDeviceProperties(at::globalContext().getTHCState(), (int)device);
}

Allocator* getCUDADeviceAllocator() {
  return at::globalContext().getTHCState()->cudaDeviceAllocator;
}

/* Handles */
cusparseHandle_t getCurrentCUDASparseHandle() {
  return THCState_getCurrentSparseHandle(at::globalContext().getTHCState());
}

cublasHandle_t getCurrentCUDABlasHandle() {
  return THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
}

#ifdef AT_ROCM_ENABLED()

namespace {

struct HIPGuardImplMasqueradingAsCUDA final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::CUDA;
  HIPGuardImplMasqueradingAsCUDA() {}
  HIPGuardImplMasqueradingAsCUDA(DeviceType t) {
    AT_ASSERT(t == DeviceType::CUDA);
  }
  DeviceType type() const override {
    return DeviceType::CUDA;
  }
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::CUDA);
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      C10_HIP_CHECK(hipSetDevice(d.index()));
    }
    return old_device;
  }
  Device getDevice() const override {
    int device;
    C10_HIP_CHECK(hipGetDevice(&device));
    return Device(DeviceType::CUDA, device);
  }
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::CUDA);
    C10_HIP_CHECK(hipSetDevice(d.index()));
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    hipSetDevice(d.index());
  }
  Stream getStream(Device d) const noexcept override {
    return getCurrentHIPStream().unwrap();
  }
  Stream exchangeStream(Stream s) const noexcept override {
    HIPStream cs(s);
    auto old_stream = getCurrentHIPStream(s.device().index());
    setCurrentHIPStream(cs);
    return old_stream.unwrap();
  }
};

}

// THIS IS A MASSIVE HACK
C10_REGISTER_GUARD_IMPL(CUDA, HIPGuardImplMasqueradingAsCUDA);

#endif

} // namespace cuda

} // namespace at
