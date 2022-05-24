//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSDevice.h>

namespace at {
namespace mps {

static std::unique_ptr<MPSDevice> mps_device;
static std::once_flag mpsdev_init;

MPSDevice* MPSDevice::getInstance() {
  std::call_once(mpsdev_init, [] {
      mps_device = std::unique_ptr<MPSDevice>(new MPSDevice());
  });
  return mps_device.get();
}

MPSDevice::~MPSDevice() {
  [_mtl_device release];
  _mtl_device = nil;
}

MPSDevice::MPSDevice(): _mtl_device(nil) {
  NSArray* devices = [MTLCopyAllDevices() autorelease];
  for (unsigned long i = 0 ; i < [devices count] ; i++) {
    id<MTLDevice>  device = devices[i];
    if(![device isLowPower]) { // exclude Intel GPUs
      _mtl_device = [device retain];
      break;
    }
  }
  assert(_mtl_device);
}

at::Allocator* getMPSSharedAllocator();
at::Allocator* GetMPSAllocator(bool useSharedAllocator) {
  return useSharedAllocator ? getMPSSharedAllocator() : GetAllocator(DeviceType::MPS);
}

bool is_available() {
  return MPSDevice::getInstance()->device() != nil;
}

} // namespace mps
} // namespace at
