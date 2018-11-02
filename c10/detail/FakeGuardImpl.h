#include <c10/detail/DeviceGuardImplInterface.h>

namespace c10 {
namespace detail {

/**
 * A fake implementation of DeviceGuardImplInterface suitable for testing.
 * The current device is modeled as a mutable field in the guard implementation
 * class.  See DeviceGuard_test.cpp for an example use.
 */
template <DeviceType T>
struct FakeGuardImpl final : public DeviceGuardImplInterface {
  DeviceType type() const override {
    return T;
  }
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == type());
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
    current_device_ = d.index();
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    current_device_ = d.index();
  }
  // Convenience methods for testing
  static DeviceIndex getDeviceIndex() {
    return current_device_;
  }
  static void setDeviceIndex(DeviceIndex i) {
    AT_ASSERT(i >= 0);
    current_device_ = i;
  }
private:
  thread_local static DeviceIndex current_device_;
};

template <DeviceType T>
thread_local DeviceIndex FakeGuardImpl<T>::current_device_ = 0;

}} // namespace c10::detail
