#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <array>

namespace c10::impl {

// FakeGuardImpl is hardcoded to have eight devices.  Not for
// any good reason, just to simplify code.
constexpr DeviceIndex kFakeGuardImplMaxDevices = 8;

/**
 * A fake implementation of DeviceGuardImplInterface suitable for testing.
 * The current device is modeled as a mutable field in the guard implementation
 * class.  See DeviceGuard_test.cpp for an example use.
 */
template <DeviceType T>
struct FakeGuardImpl final : public DeviceGuardImplInterface {
  static constexpr DeviceType static_type = T;
  // Runtime device type is not used
  FakeGuardImpl(DeviceType) {}
  FakeGuardImpl() = default;
  DeviceType type() const override {
    return T;
  }
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == type());
    AT_ASSERT(d.index() < kFakeGuardImplMaxDevices);
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      current_device_ = d.index();
    }
    return old_device;
  }
  Device getDevice() const override {
    return Device(type(), current_device_);
  }
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == type());
    AT_ASSERT(d.index() >= 0);
    AT_ASSERT(d.index() < kFakeGuardImplMaxDevices);
    current_device_ = d.index();
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    current_device_ = d.index();
  }
  Stream getStream(Device d) const noexcept override {
    return Stream(Stream::UNSAFE, d, current_streams_[d.index()]);
  }
  Stream exchangeStream(Stream s) const noexcept override {
    auto old_id = current_streams_[s.device_index()];
    current_streams_[s.device_index()] = s.id();
    return Stream(Stream::UNSAFE, s.device(), old_id);
  }
  DeviceIndex deviceCount() const noexcept override {
    return kFakeGuardImplMaxDevices;
  }

  // Event-related functions
  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {}
  void block(void* event, const Stream& stream) const override {}
  bool queryEvent(void* event) const override {
    return true;
  }
  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {}

  // Convenience methods for testing
  static DeviceIndex getDeviceIndex() {
    return current_device_;
  }
  static void setDeviceIndex(DeviceIndex i) {
    AT_ASSERT(i >= 0);
    AT_ASSERT(i < kFakeGuardImplMaxDevices);
    current_device_ = i;
  }
  static StreamId getCurrentStreamIdFor(DeviceIndex i) {
    return current_streams_.at(i);
  }
  static void resetStreams() {
    current_streams_.fill(0);
  }

 private:
  thread_local static DeviceIndex current_device_;
  thread_local static std::array<StreamId, kFakeGuardImplMaxDevices>
      current_streams_;
};

template <DeviceType T>
thread_local DeviceIndex FakeGuardImpl<T>::current_device_ = 0;

template <DeviceType T>
thread_local std::array<StreamId, kFakeGuardImplMaxDevices>
    FakeGuardImpl<T>::current_streams_ = {0, 0, 0, 0, 0, 0, 0, 0};

} // namespace c10::impl
