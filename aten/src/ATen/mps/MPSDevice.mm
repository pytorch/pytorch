//  Copyright © 2022 Apple Inc.

#include <c10/util/CallOnce.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/IndexKernels.h>

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

id<MTLFunction> MPSDevice::metalFunction(const std::string& kernel, MTLFunctionConstantValues* constantValues) {
  assert(_mtl_device);
  NSError* error = nil;
  if (!_mtl_indexing_library) {
    _mtl_indexing_library = [_mtl_device newLibraryWithSource: [NSString stringWithCString: mps::indexing_metal_shaders encoding:NSASCIIStringEncoding]
                                                      options: nil
                                                        error: &error];
    TORCH_CHECK(_mtl_indexing_library, "Failed to create indexing library, error: ", [[error description] UTF8String]);
  }

  id<MTLFunction> indexFunction = [_mtl_indexing_library newFunctionWithName: [NSString stringWithUTF8String:kernel.c_str()]
                                                              constantValues: constantValues
                                                                       error: &error];
  TORCH_CHECK(indexFunction, "Failed to create specialized function state object: ", kernel, ", error: ", [[error description] UTF8String]);

  return indexFunction;
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
  _mtl_indexing_library = nil;
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
