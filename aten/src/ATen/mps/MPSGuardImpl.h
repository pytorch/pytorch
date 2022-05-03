//  Copyright © 2022 Apple Inc.

#pragma once
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <ATen/mps/MPSStream.h>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#include <ATen/Tensor.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <sys/_types/_size_t.h>
#include <memory>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/intrusive_ptr.h>


namespace at {
namespace mps {

struct TORCH_API MPSGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::MPS;

  // constructor
  MPSGuardImpl() {}
  explicit MPSGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::MPS);
  }

  // returns the type
  DeviceType type() const override {
    return DeviceType::MPS;
  }

  Device exchangeDevice(Device d) const override {
    return Device(DeviceType::MPS, 0);
  }

  Device getDevice() const override {
    return Device(DeviceType::MPS, 0);
  }

  c10::optional<Device> uncheckedGetDevice() const noexcept {
    return Device(DeviceType::MPS, 0);
  }

  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_mps());
  }

  void uncheckedSetDevice(Device d) const noexcept override {
    // TODO: Currently setting only device 0
  }

  Stream getStream(Device d) const noexcept override {
    return Stream(Stream::DEFAULT, Device(DeviceType::MPS, 0));
  }

  Stream getDefaultStream(Device d) const override {
    return Stream(Stream::DEFAULT, Device(DeviceType::MPS, 0));
  }

  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    return Stream(Stream::DEFAULT, Device(DeviceType::MPS, 0));
  }
  DeviceIndex deviceCount() const noexcept override {
    //TODO: extend it for multi-device case
    return 1;
  }

  // Event-related functions
  void createEvent(
    mpsEvent_t* event,
    const EventFlag flag) const;

  void destroyEvent(
    void* event,
    const DeviceIndex device_index) const noexcept override;

  void record(
    void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const EventFlag flag) const override;

  void block(
    void* event,
    const Stream& stream) const override;

  bool queryEvent(void* event) const override;

};

C10_REGISTER_GUARD_IMPL(MPS, MPSGuardImpl);

}} // namespace at::mps
