#include "../include/openreg.h"

namespace {
//
constexpr int DEVICE_COUNT = 4;
//
thread_local int gCurrentDevice = 0;
} // namespace

orError_t orGetDeviceCount(int* count) {
  if (!count) {
    return orErrorUnknown;
  }

  *count = DEVICE_COUNT;
  return orSuccess;
}

orError_t orSetDevice(int device) {
  if (device < 0 || device >= DEVICE_COUNT) {
    return orErrorUnknown;
  }

  gCurrentDevice = device;
  return orSuccess;
}

orError_t orGetDevice(int* device) {
  if (!device) {
    return orErrorUnknown;
  }

  *device = gCurrentDevice;
  return orSuccess;
}
