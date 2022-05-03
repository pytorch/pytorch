//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSDevice.h>

namespace at {
namespace mps {

MPSDevice* MPSDevice::_device = nullptr;

MPSDevice* MPSDevice::getInstance() {
  if (_device == nullptr) {
    _device = new MPSDevice();
  }
  return _device;
}

MPSDevice::~MPSDevice() {
  [_mtl_device release];
  _mtl_device = nil;
}

MPSDevice::MPSDevice() {
  NSArray* devices = MTLCopyAllDevices();
  bool allowIntelGPUs = false;
  for (unsigned long i = 0 ; i < [devices count] ; i++) {
    id<MTLDevice>  device = devices[i];
    if(![device isLowPower]) { // exclude Intel GPUs
      _mtl_device = device;
      break;
    }
  }
  assert(_mtl_device);
}

at::Allocator* getMPSSharedAllocator();
at::Allocator* GetMPSAllocator(bool useSharedAllocator) {
  return useSharedAllocator ? getMPSSharedAllocator() : GetAllocator(DeviceType::MPS);
}

} // namespace mps
} // namespace at
