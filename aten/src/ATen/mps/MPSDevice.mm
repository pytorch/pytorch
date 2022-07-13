//  Copyright © 2022 Apple Inc.

#include <c10/util/CallOnce.h>

#include <ATen/mps/MPSDevice.h>

namespace at {
namespace mps {

static std::unique_ptr<MPSDevice> mps_device;
static c10::once_flag mpsdev_init;

MPSDevice* MPSDevice::getInstance() {
  c10::call_once(mpsdev_init, [] {
      mps_device = std::unique_ptr<MPSDevice>(new MPSDevice());
  });
  return mps_device.get();
}

MPSDevice::~MPSDevice() {
  [_mtl_device release];
  _mtl_device = nil;
}

MPSDevice::MPSDevice(): _mtl_device(nil) {
  // Check that MacOS 12.3+ version of MPS framework is available
  // Create the MPSGraph and check method introduced in 12.3+
  // which is used by MPS backend.
  id mpsCD = NSClassFromString(@"MPSGraph");
  if ([mpsCD instancesRespondToSelector:@selector(LSTMWithSourceTensor:
                                                       recurrentWeight:
                                                           inputWeight:
                                                                  bias:
                                                             initState:
                                                              initCell:
                                                            descriptor:
                                                                  name:)] == NO) {
    return;
  }
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
at::Allocator* getMPSStaticAllocator();
at::Allocator* GetMPSAllocator(bool useSharedAllocator) {
  return useSharedAllocator ? getMPSSharedAllocator() : getMPSStaticAllocator();
}

bool is_available() {
  return MPSDevice::getInstance()->device() != nil;
}

} // namespace mps
} // namespace at
