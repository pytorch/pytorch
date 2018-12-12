#pragma once

#include <ATen/cuda/CUDAConfig.h>

// The includes of CUDAGuard.h
#include <c10/cuda/impl/CUDAGuardImpl.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/DeviceType.h>
#include <c10/impl/InlineDeviceGuard.h>
#include <c10/impl/InlineStreamGuard.h>

#if AT_ROCM_ENABLED()
#include <c10/hip/impl/HIPGuardImpl.h>

// Use of c10::hip namespace here makes hipification easier
namespace c10 { namespace hip {

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
using OptionalHIPGuardMasqueradingAsCUDA = c10::impl::InlineOptionalDeviceGuard<HIPGuardImplMasqueradingAsCUDA>;
using HIPStreamGuardMasqueradingAsCUDA = c10::impl::InlineStreamGuard<HIPGuardImplMasqueradingAsCUDA>;
using OptionalHIPStreamGuardMasqueradingAsCUDA = c10::impl::InlineOptionalStreamGuard<HIPGuardImplMasqueradingAsCUDA>;

}} // namespace c10::hip

#endif
