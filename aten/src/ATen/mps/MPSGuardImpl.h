//  Copyright © 2022 Apple Inc.

#pragma once
#include <ATen/Context.h>
#include <ATen/mps/MPSEvent.h>
#include <ATen/mps/MPSStream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#include <ATen/Tensor.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <sys/_types/_size_t.h>
#include <memory>

namespace at::mps {

typedef MPSEvent* mpsEvent_t;

// TODO: Move the MPSGuardImpl to inherit from NoOpDeviceGuardImpl
// https://github.com/pytorch/pytorch/issues/77170
struct TORCH_API MPSGuardImpl final
    : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::MPS;

  // constructor
  MPSGuardImpl() {}
  explicit MPSGuardImpl(c10::DeviceType t) {
    TORCH_CHECK(
        t == DeviceType::MPS,
        "MPSGuardImpl initialized with non-MPS DeviceType: ",
        t);
  }

  // returns the type
  c10::DeviceType type() const override {
    return c10::DeviceType::MPS;
  }

  Device exchangeDevice(Device d) const override {
    return Device(c10::DeviceType::MPS, 0);
  }

  Device getDevice() const override {
    return Device(c10::DeviceType::MPS, 0);
  }

  std::optional<Device> uncheckedGetDevice() const noexcept {
    return Device(c10::DeviceType::MPS, 0);
  }

  void setDevice(Device d) const override {
    TORCH_CHECK(d.is_mps(), "Expected a MPS device, but got ", d);
  }

  void uncheckedSetDevice(Device d) const noexcept override {
    // TODO: Currently setting only device 0
  }

  Stream getStream(Device d) const override {
    return Stream(Stream::DEFAULT, Device(c10::DeviceType::MPS, 0));
  }

  Stream getNewStream(Device, int priority = 0) const override {
    (void)priority;
    return Stream(Stream::DEFAULT, Device(c10::DeviceType::MPS, 0));
  }

  Stream getDefaultStream(Device d) const override {
    return Stream(Stream::DEFAULT, Device(c10::DeviceType::MPS, 0));
  }

  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const override {
    return Stream(Stream::DEFAULT, Device(c10::DeviceType::MPS, 0));
  }
  DeviceIndex deviceCount() const noexcept override {
    if (at::hasMPS()) {
      // TODO: extend it for multi-device case
      return 1;
    } else {
      return 0;
    }
  }

  // Event-related functions
  void createEvent(mpsEvent_t* event, const EventFlag flag) const;

  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override;

  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override;

  void block(void* event, const Stream& stream) const override;

  bool queryEvent(void* event) const override;

  void synchronizeEvent(void* event) const override;

  double elapsedTime(void* event1, void* event2, const DeviceIndex device_index)
      const override;

  void synchronizeDevice(const DeviceIndex device_index) const override;
};

/// A variant of OptionalDeviceGuard that is specialized for MPS.
struct OptionalMPSGuard {
  explicit OptionalMPSGuard() : guard_() {}

  explicit OptionalMPSGuard(std::optional<Device> device_opt)
      : guard_(device_opt) {}

  /// Set the current MPS device to the passed device index, if it is not
  /// nullopt
  explicit OptionalMPSGuard(std::optional<DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalMPSGuard(const OptionalMPSGuard&) = delete;
  OptionalMPSGuard& operator=(const OptionalMPSGuard&) = delete;
  OptionalMPSGuard(OptionalMPSGuard&& other) = delete;
  OptionalMPSGuard& operator=(OptionalMPSGuard&& other) = delete;

  /// Sets the MPS device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a MPS
  /// device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the MPS device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a MPS device.
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the MPS device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  std::optional<Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  std::optional<Device> current_device() const {
    return guard_.current_device();
  }

  /// Restore the original MPS device, resetting this guard to uninitialized
  /// state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<MPSGuardImpl> guard_;
};

C10_REGISTER_GUARD_IMPL(MPS, MPSGuardImpl)

} // namespace at::mps
