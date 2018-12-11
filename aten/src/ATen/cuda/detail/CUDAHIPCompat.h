#pragma once

#include <ATen/cuda/CUDAConfig.h>

// Here just to make sure any transitive includes also show up
#include <c10/cuda/CUDAGuard.h>

#if AT_ROCM_ENABLED()
#include <c10/hip/impl/HIPGuardImpl.h>

namespace at { namespace cuda { namespace detail {

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

using HIPGuardMasqueradingAsCUDA = c10::impl::InlineDeviceGuard<HIPGuardImplMasqueradingAsCUDA>;
using OptionalHIPGuardMasqueradingAsCUDA = c10::impl::InlineDeviceGuard<OptionalHIPGuardImplMasqueradingAsCUDA>;
using HIPStreamGuardMasqueradingAsCUDA = c10::impl::InlineStreamGuard<HIPGuardImplMasqueradingAsCUDA>;
using OptionalHIPStreamGuardMasqueradingAsCUDA = c10::impl::InlineOptionalStreamGuard<HIPGuardImplMasqueradingAsCUDA>;

}}} // namespace at::cuda::detail

#endif
