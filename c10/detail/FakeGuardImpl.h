#pragma once

#include <c10/detail/DeviceGuardImplInterface.h>

#include <array>

namespace c10 {
namespace detail {

// FakeGuardImpl is hardcoded to have eight devices.  Not for
// any good reason, just to simplify code.
constexpr size_t kFakeGuardImplMaxDevices = 8;

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
  FakeGuardImpl() {}
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
    return Stream(d, current_streams_[d.index()]);
  }
  Stream exchangeStream(Stream s) const noexcept override {
    auto old_id = current_streams_[s.device_index()];
    current_streams_[s.device_index()] = s.id();
    return Stream(s.device(), old_id);
  }
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
  thread_local static std::array<StreamId, kFakeGuardImplMaxDevices> current_streams_;
};

template <DeviceType T>
thread_local DeviceIndex FakeGuardImpl<T>::current_device_ = 0;

template <DeviceType T>
constexpr DeviceType FakeGuardImpl<T>::static_type;

template <DeviceType T>
thread_local std::array<StreamId, kFakeGuardImplMaxDevices> FakeGuardImpl<T>::current_streams_ = {0,0,0,0,0,0,0,0};


}} // namespace c10::detail
