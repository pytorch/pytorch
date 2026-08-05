#include <c10/util/Exception.h>
#include <include/openreg.h>

#include "OpenRegException.h"
#include "OpenRegFunctions.h"

namespace c10::openreg {
namespace {

c10::DeviceIndex checked_device_index_from_int(int device) {
  TORCH_CHECK(
      device >= 0 && device < c10::Device::MAX_NUM_DEVICES,
      "OpenReg device index ",
      device,
      " is out of range for DeviceIndex [0, ",
      c10::Device::MAX_NUM_DEVICES - 1,
      "]");
  return static_cast<c10::DeviceIndex>(device);
}

} // namespace

orError_t GetDeviceCount(int* dev_count) {
  return orGetDeviceCount(dev_count);
}

orError_t GetDevice(DeviceIndex* device) {
  int tmp_device = -1;
  auto err = orGetDevice(&tmp_device);
  if (err == orSuccess) {
    *device = checked_device_index_from_int(tmp_device);
  }
  return err;
}
// LITERALINCLUDE START: OPENREG SetDevice FUNCTION
orError_t SetDevice(DeviceIndex device) {
  int cur_device = -1;
  OPENREG_CHECK(orGetDevice(&cur_device));
  if (device == cur_device) {
    return orSuccess;
  }
  return orSetDevice(device);
}
// LITERALINCLUDE END: OPENREG SetDevice FUNCTION

int device_count_impl() {
  int count = 0;
  GetDeviceCount(&count);
  return count;
}

OPENREG_EXPORT DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      auto result = device_count_impl();
      TORCH_CHECK(
          result <= c10::Device::MAX_NUM_DEVICES,
          "Too many devices, DeviceIndex overflowed");
      return result;
    } catch (const Error& ex) {
      // We don't want to fail, but still log the warning
      // msg() returns the message without the stack trace
      TORCH_WARN("Device initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<DeviceIndex>(count);
}

OPENREG_EXPORT DeviceIndex current_device() {
  DeviceIndex cur_device = -1;
  OPENREG_CHECK(GetDevice(&cur_device));
  return cur_device;
}

// LITERALINCLUDE START: OPENREG set_device FUNCTION
OPENREG_EXPORT void set_device(DeviceIndex device) {
  check_device_index(device);
  OPENREG_CHECK(SetDevice(device));
}
// LITERALINCLUDE END: OPENREG set_device FUNCTION

OPENREG_EXPORT DeviceIndex ExchangeDevice(DeviceIndex device) {
  check_device_index(device);
  int current_device = -1;
  OPENREG_CHECK(orGetDevice(&current_device));

  if (device != current_device) {
    OPENREG_CHECK(orSetDevice(device));
  }

  return checked_device_index_from_int(current_device);
}

OPENREG_EXPORT DeviceIndex maybe_exchange_device(DeviceIndex to_device) {
  check_device_index(to_device);
  return ExchangeDevice(to_device);
}
} // namespace c10::openreg
