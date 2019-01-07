#pragma once

#include <ATen/hip/HIPConfig.h>

// The includes of HIPGuard.h
#include <c10/hip/impl/HIPGuardImpl.h>
#include <c10/hip/HIPMacros.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include <c10/hip/impl/HIPGuardImpl.h>

// Use of c10::hip namespace here makes hipification easier, because
// I don't have to also fix namespaces.  Sorry!
namespace c10 { namespace hip {

// HIPGuardImplMasqueradingAsCUDA is like a HIPGuardImpl, but
// it reports its DeviceType as CUDA (e.g., type() returns CUDA,
// getDevice() reports the current HIP device as a CUDA device.)
// We can't directly use HIPGuardImpl, since it (piously) requires
// the DeviceType to be HIP.
//
// This is necessary for PyTorch at the moment, which is implemented
// by pretending that CUDA is actually HIP.  Eventually, we want
// to make PyTorch treat HIP as a separate DeviceType, and then we
// can delete this class.
//
// Also, note that the cpp file associated with this also *overwrites*
// the entry in the DeviceGuardImpl registry for CUDA with this HIP
// implementation.
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
  DeviceIndex deviceCount() const override {
    int deviceCnt;
    C10_HIP_CHECK(hipGetDeviceCount(&deviceCnt));
    return deviceCnt;
  }
};

// All of the guards which have HIPGuardImpl burned in need to also have
// variants using HIPGuardImplMasqueradingAsCUDA.
using HIPGuardMasqueradingAsCUDA = c10::impl::InlineDeviceGuard<HIPGuardImplMasqueradingAsCUDA>;
using OptionalHIPGuardMasqueradingAsCUDA = c10::impl::InlineOptionalDeviceGuard<HIPGuardImplMasqueradingAsCUDA>;
using HIPStreamGuardMasqueradingAsCUDA = c10::impl::InlineStreamGuard<HIPGuardImplMasqueradingAsCUDA>;
using OptionalHIPStreamGuardMasqueradingAsCUDA = c10::impl::InlineOptionalStreamGuard<HIPGuardImplMasqueradingAsCUDA>;

}} // namespace c10::hip
