//  Copyright © 2022 Apple Inc.

#include <c10/util/CallOnce.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/IndexKernels.h>

namespace at {
namespace mps {

static std::unique_ptr<MPSDevice> mps_device;
static c10::once_flag mpsdev_init;

static inline MTLLanguageVersion getMetalLanguageVersion(const id<MTLDevice>& device) {
  // MPS Advanced Indexing needs at least Metal 2.0 (support for Argument Buffers and function constants)
  // host_name attribute needs at least Metal 2.2
  MTLLanguageVersion languageVersion = MTLLanguageVersion2_2;

  TORCH_CHECK([device supportsFamily:MTLGPUFamilyMac2], "Missing Metal support for MTLGPUFamilyMac2");
  return languageVersion;
}

MPSDevice* MPSDevice::getInstance() {
  c10::call_once(mpsdev_init, [] {
      mps_device = std::unique_ptr<MPSDevice>(new MPSDevice());
  });
  return mps_device.get();
}

id<MTLFunction> MPSDevice::metalIndexingFunction(const std::string& kernel, MTLFunctionConstantValues* constantValues) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_mtl_device);
  NSError* error = nil;
  if (!_mtl_indexing_library) {
    MTLCompileOptions *options = [MTLCompileOptions new];
    [options setLanguageVersion: getMetalLanguageVersion(_mtl_device)];
    [options setFastMathEnabled: YES];
    _mtl_indexing_library = [_mtl_device newLibraryWithSource: [NSString stringWithCString: mps::indexing_metal_shaders encoding:NSASCIIStringEncoding]
                                                      options: options
                                                        error: &error];
    TORCH_CHECK(_mtl_indexing_library, "Failed to create indexing library, error: ", [[error description] UTF8String]);
  }

  id<MTLFunction> indexFunction = nil;
  if (constantValues) {
    indexFunction = [[_mtl_indexing_library newFunctionWithName: [NSString stringWithUTF8String: kernel.c_str()]
                                                constantValues: constantValues
                                                         error: &error] autorelease];
  } else {
    indexFunction = [[_mtl_indexing_library newFunctionWithName: [NSString stringWithUTF8String: kernel.c_str()]] autorelease];
  }

  TORCH_CHECK(indexFunction, "Failed to create specialized function state object: ", kernel, ", error: ", [[error description] UTF8String]);

  return indexFunction;
}

MPSDevice::~MPSDevice() {
  [_mtl_device release];
  [_mtl_indexing_library release];
  _mtl_device = nil;
  _mtl_indexing_library = nil;
}

MPSDevice::MPSDevice(): _mtl_device(nil), _mtl_indexing_library(nil)  {
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
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_mtl_device);
}

at::Allocator* getMPSSharedAllocator();
at::Allocator* getMPSPrivateAllocator();
at::Allocator* GetMPSAllocator(bool useSharedAllocator) {
  return useSharedAllocator ? getMPSSharedAllocator() : getMPSPrivateAllocator();
}

bool is_available() {
  return MPSDevice::getInstance()->device() != nil;
}

} // namespace mps
} // namespace at
