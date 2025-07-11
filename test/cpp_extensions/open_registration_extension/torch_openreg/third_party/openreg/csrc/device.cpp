#include <include/openreg.h>

namespace {
// Total device numbers
constexpr int DEVICE_COUNT = 2;
// Current device index
thread_local int gCurrentDevice = 0;
} // namespace

orError_t orGetDeviceCount(int* count) {
  if (!count) {
    return orErrorUnknown;
  }

  *count = DEVICE_COUNT;
  return orSuccess;
}

orError_t orGetDevice(int* device) {
  if (!device) {
    return orErrorUnknown;
  }

  *device = gCurrentDevice;
  return orSuccess;
}

orError_t orSetDevice(int device) {
  if (device < 0 || device >= DEVICE_COUNT) {
    return orErrorUnknown;
  }

  gCurrentDevice = device;
  return orSuccess;
}
