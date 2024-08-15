//  Copyright © 2022 Apple Inc.

#include <c10/util/CallOnce.h>

#include <ATen/mps/IndexKernels.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

namespace at::mps {

static std::unique_ptr<MPSDevice> mps_device;
static c10::once_flag mpsdev_init;

static inline MTLLanguageVersion getMetalLanguageVersion(const id<MTLDevice>& device) {
  // MPS Advanced Indexing needs at least Metal 2.0 (support for Argument Buffers and function constants)
  // host_name attribute needs at least Metal 2.2 and ulong needs Metal 2.3 (supported on MacOS 11+
  TORCH_CHECK([device supportsFamily:MTLGPUFamilyMac2], "Missing Metal support for MTLGPUFamilyMac2");
  return MTLLanguageVersion3_0;
}

MPSDevice* MPSDevice::getInstance() {
  c10::call_once(mpsdev_init, [] { mps_device = std::unique_ptr<MPSDevice>(new MPSDevice()); });
  return mps_device.get();
}

id<MTLLibrary> MPSDevice::getMetalIndexingLibrary() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_mtl_device);
  NSError* error = nil;
  if (!_mtl_indexing_library) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    [options setLanguageVersion:getMetalLanguageVersion(_mtl_device)];
    [options setFastMathEnabled:YES];
    _mtl_indexing_library = [_mtl_device newLibraryWithSource:[NSString stringWithCString:mps::indexing_metal_shaders
                                                                                 encoding:NSASCIIStringEncoding]
                                                      options:options
                                                        error:&error];
    TORCH_CHECK(_mtl_indexing_library, "Failed to create indexing library, error: ", [[error description] UTF8String]);
  }
  return _mtl_indexing_library;
}

id<MTLComputePipelineState> MPSDevice::metalIndexingPSO(const std::string& kernel) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_mtl_device);
  NSError* error = nil;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
  id<MTLLibrary> indexing_lib = getMetalIndexingLibrary();
  id<MTLComputePipelineState> state = psoCache[kernel];
  if (state) {
    return state;
  }

  id<MTLFunction> indexFunction =
      [[indexing_lib newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]] autorelease];
  TORCH_CHECK(indexFunction, "Can't find function ", kernel);

  state = [_mtl_device newComputePipelineStateWithFunction:indexFunction error:&error];
  TORCH_CHECK(state, error.localizedDescription.UTF8String);
  psoCache[kernel] = state;
  return state;
}

MPSDevice::~MPSDevice() {
  [_mtl_device release];
  [_mtl_indexing_library release];
  _mtl_device = nil;
  _mtl_indexing_library = nil;
}

MPSDevice::MPSDevice() : _mtl_device(nil), _mtl_indexing_library(nil) {
  // Check that MacOS 13.0+ version of MPS framework is available
  // Create the MPSGraph and check method introduced in 13.0
  // which is used by MPS backend.
  id mpsCD = NSClassFromString(@"MPSGraph");

  if ([mpsCD instancesRespondToSelector:@selector(cumulativeSumWithTensor:axis:name:)] == NO) {
    return;
  }

  NSArray* devices = [MTLCopyAllDevices() autorelease];
  for (unsigned long i = 0; i < [devices count]; i++) {
    id<MTLDevice> device = devices[i];
    if ([device isLowPower]) { // exclude Intel GPUs
      continue;
    }
    if (![device supportsFamily:MTLGPUFamilyMac2]) {
      // Exclude devices that does not support Metal 2.0
      // Virtualised MPS device on MacOS 12.6 should fail this check
      TORCH_WARN("Skipping device ", [[device name] UTF8String], " that does not support Metal 2.0");
      continue;
    }
    _mtl_device = [device retain];
    break;
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_mtl_device);
}

bool MPSDevice::isMacOS13Plus(MacOSVersion version) const {
  auto is_os_version_at_least = [](int major, int minor) {
    @autoreleasepool {
      NSProcessInfo* processInfo = [[NSProcessInfo alloc] init];
      return [processInfo
          isOperatingSystemAtLeastVersion:{.majorVersion = major, .minorVersion = minor, .patchVersion = 0}];
    }
  };
  static bool _macos_13_1_plus = is_os_version_at_least(13, 1);
  static bool _macos_13_2_plus = is_os_version_at_least(13, 2);
  static bool _macos_13_3_plus = is_os_version_at_least(13, 3);
  static bool _macos_14_0_plus = is_os_version_at_least(14, 0);
  static bool _macos_14_4_plus = is_os_version_at_least(14, 4);
  static bool _macos_15_0_plus = is_os_version_at_least(15, 0);

  switch (version) {
    case MacOSVersion::MACOS_VER_13_1_PLUS:
      return _macos_13_1_plus;
    case MacOSVersion::MACOS_VER_13_2_PLUS:
      return _macos_13_2_plus;
    case MacOSVersion::MACOS_VER_13_3_PLUS:
      return _macos_13_3_plus;
    case MacOSVersion::MACOS_VER_14_0_PLUS:
      return _macos_14_0_plus;
    case MacOSVersion::MACOS_VER_14_4_PLUS:
      return _macos_14_4_plus;
    case MacOSVersion::MACOS_VER_15_0_PLUS:
      return _macos_15_0_plus;
    default:
      return false;
  }
}

at::Allocator* GetMPSAllocator(bool useSharedAllocator) {
  return getIMPSAllocator(useSharedAllocator);
}

bool is_available() {
  return MPSDevice::getInstance()->device() != nil;
}

bool is_macos_13_or_newer(MacOSVersion version) {
  return MPSDevice::getInstance()->isMacOS13Plus(version);
}

} // namespace at::mps
