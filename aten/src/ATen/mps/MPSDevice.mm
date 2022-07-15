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
  MTLLanguageVersion languageVersion;

#if defined(__MAC_13_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_13_0
  languageVersion = MTLLanguageVersion3_0;
#elif defined(__MAC_12_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_12_0
  languageVersion = MTLLanguageVersion2_4;
#elif defined(__MAC_11_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_11_0
  languageVersion = MTLLanguageVersion2_3;
#elif defined(__MAC_10_15) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_15
  languageVersion = MTLLanguageVersion2_2;
#elif defined(__MAC_10_14) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_14
  languageVersion = MTLLanguageVersion2_1;
#elif defined(__MAC_10_13) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_13
    languageVersion = MTLLanguageVersion2_0;
#elif
  #error "Metal is not available on the current platform."
#endif

  TORCH_CHECK([device supportsFamily:MTLGPUFamilyMac2], "Missing Metal support for MTLGPUFamilyMac2");
  return languageVersion;
}

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
    MTLCompileOptions *options = [MTLCompileOptions new];
    [options setLanguageVersion: getMetalLanguageVersion(_mtl_device)];
    [options setFastMathEnabled: YES];
    _mtl_indexing_library = [_mtl_device newLibraryWithSource: [NSString stringWithCString: mps::indexing_metal_shaders encoding:NSASCIIStringEncoding]
                                                      options: options
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
