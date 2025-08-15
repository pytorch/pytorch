#include <include/openreg.h>

#include "OpenRegFunctions.h"

namespace c10::openreg {

orError_t GetDeviceCount(int* dev_count) {
  return orGetDeviceCount(dev_count);
}

orError_t GetDevice(c10::DeviceIndex* device) {
  int tmp_device = -1;
  auto err = orGetDevice(&tmp_device);
  *device = static_cast<c10::DeviceIndex>(tmp_device);
  return err;
}

orError_t SetDevice(c10::DeviceIndex device) {
  int cur_device = -1;
  orGetDevice(&cur_device);
  if (device == cur_device) {
    return orSuccess;
  }
  return orSetDevice(device);
}

int device_count_impl() {
  int count = 0;
  GetDeviceCount(&count);
  return count;
}

c10::DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      auto result = device_count_impl();
      TORCH_INTERNAL_ASSERT(
          result <= std::numeric_limits<c10::DeviceIndex>::max(),
          "Too many devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // We don't want to fail, but still log the warning
      // msg() returns the message without the stack trace
      TORCH_WARN("Device initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<c10::DeviceIndex>(count);
}

c10::DeviceIndex current_device() {
  c10::DeviceIndex cur_device = -1;
  GetDevice(&cur_device);
  return cur_device;
}

void set_device(c10::DeviceIndex device) {
  SetDevice(device);
}

DeviceIndex ExchangeDevice(DeviceIndex device) {
  int current_device = -1;
  orGetDevice(&current_device);

  if (device != current_device) {
    orSetDevice(device);
  }

  return current_device;
}

} // namespace c10::openreg
