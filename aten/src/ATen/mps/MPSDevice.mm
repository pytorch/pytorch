//  Copyright © 2022 Apple Inc.

#include <ATen/mps/IndexKernels.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/MPSGraphSequoiaOps.h>

namespace at::mps {

static inline MTLLanguageVersion getMetalLanguageVersion(const id<MTLDevice>& device) {
  // MPS Advanced Indexing needs at least Metal 2.0 (support for Argument Buffers and function constants)
  // host_name attribute needs at least Metal 2.2 and ulong needs Metal 2.3 (supported on MacOS 11+
  TORCH_CHECK([device supportsFamily:MTLGPUFamilyMac2], "Missing Metal support for MTLGPUFamilyMac2");
  return MTLLanguageVersion3_0;
}

MPSDevice* MPSDevice::getInstance() {
  static MPSDevice mps_device;
  return &mps_device;
}

MPSDevice::~MPSDevice() {
  [_mtl_device release];
  _mtl_device = nil;
}

MPSDevice::MPSDevice() : _mtl_device(nil) {
  // Check that MacOS 13.0+ version of MPS framework is available
  // Create the MPSGraph and check method introduced in 14.0
  // which is used by MPS backend.
  id mpsCD = NSClassFromString(@"MPSGraph");

  if ([mpsCD instancesRespondToSelector:@selector(HermiteanToRealFFTWithTensor:axes:descriptor:name:)] == NO) {
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
      NSProcessInfo* processInfo = [[NSProcessInfo new] autorelease];
      return [processInfo
          isOperatingSystemAtLeastVersion:{.majorVersion = major, .minorVersion = minor, .patchVersion = 0}];
    }
  };
  static bool _macos_14_4_plus = is_os_version_at_least(14, 4);
  static bool _macos_15_0_plus = is_os_version_at_least(15, 0);
  static bool _macos_15_1_plus = is_os_version_at_least(15, 1);
  static bool _macos_15_2_plus = is_os_version_at_least(15, 2);
  static bool _macos_26_0_plus = is_os_version_at_least(26, 0);

  switch (version) {
    case MacOSVersion::MACOS_VER_14_4_PLUS:
      return _macos_14_4_plus;
    case MacOSVersion::MACOS_VER_15_0_PLUS:
      return _macos_15_0_plus;
    case MacOSVersion::MACOS_VER_15_1_PLUS:
      return _macos_15_1_plus;
    case MacOSVersion::MACOS_VER_15_2_PLUS:
      return _macos_15_2_plus;
    case MacOSVersion::MACOS_VER_26_0_PLUS:
      return _macos_26_0_plus;
    default:
      return false;
  }
}

std::string MPSDevice::getName() const {
  @autoreleasepool {
    return [[_mtl_device name] UTF8String];
  }
}

unsigned MPSDevice::getCoreCount() const {
  io_iterator_t iterator = 0;
  io_registry_entry_t entry = 0;
  int core_count = 0;
  auto matchingDict = IOServiceMatching("AGXAccelerator");
  TORCH_INTERNAL_ASSERT(matchingDict, "Failed to create matching dict");
  const auto status = IOServiceGetMatchingServices(kIOMainPortDefault, matchingDict, &iterator);
  TORCH_INTERNAL_ASSERT(status == KERN_SUCCESS);
  while ((entry = IOIteratorNext(iterator)) != 0) {
    auto property = IORegistryEntryCreateCFProperty(entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
    auto found = CFNumberGetValue(static_cast<CFNumberRef>(property), kCFNumberIntType, &core_count);
    CFRelease(property);
    IOObjectRelease(entry);
    if (found) {
      break;
    }
  }
  IOObjectRelease(iterator);
  return core_count;
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
