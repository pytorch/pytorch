//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/MPSGraphSequoiaOps.h>
#include <c10/util/env.h>

namespace at::mps {

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
  static bool _macos_26_2_plus = is_os_version_at_least(26, 2);
  static bool _macos_26_4_plus = is_os_version_at_least(26, 4);
  static bool _macos_27_0_plus = is_os_version_at_least(27, 0);

  switch (version) {
    case MacOSVersion::MACOS_14_4:
      return _macos_14_4_plus;
    case MacOSVersion::MACOS_15_0:
      return _macos_15_0_plus;
    case MacOSVersion::MACOS_15_1:
      return _macos_15_1_plus;
    case MacOSVersion::MACOS_15_2:
      return _macos_15_2_plus;
    case MacOSVersion::MACOS_26_0:
      return _macos_26_0_plus;
    case MacOSVersion::MACOS_26_2:
      return _macos_26_2_plus;
    case MacOSVersion::MACOS_26_4:
      return _macos_26_4_plus;
    case MacOSVersion::MACOS_27_0:
      return _macos_27_0_plus;
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

at::Allocator* GetMPSAllocator() {
  return getIMPSAllocator();
}
bool is_available() {
  return MPSDevice::getInstance()->device() != nil;
}

bool is_macos_at_least(MacOSVersion version) {
  return MPSDevice::getInstance()->isMacOS13Plus(version);
}

bool is_apple_family_or_newer(AppleGPUFamily family) {
  // some ops which are on MPSGraph behave differently between GPU families
  auto mtl_family = static_cast<MTLGPUFamily>(family);
  return [MPSDevice::getInstance()->device() supportsFamily:mtl_family];
}

// MetalLanguageVersion mirrors the SDK enum so that compileLibrary can hand it
// straight to setLanguageVersion:; keep the two spellings from drifting apart
static_assert(static_cast<uint32_t>(MetalLanguageVersion::METAL_3_1) == MTLLanguageVersion3_1);
static_assert(static_cast<uint32_t>(MetalLanguageVersion::METAL_3_2) == MTLLanguageVersion3_2);
static_assert(static_cast<uint32_t>(MetalLanguageVersion::METAL_4_0) == MTLLanguageVersion4_0);

MetalLanguageVersion metal_language_version() {
  static const MetalLanguageVersion rc = []() {
    if (const auto env_val = c10::utils::get_env("PYTORCH_MPS_METAL_VERSION")) {
      // MTLLanguageVersion is a closed enum, so match its spellings rather than
      // parse an arbitrary major.minor. The request is honored as-is: asking for
      // a version the host cannot compile should fail loudly, not fall back.
      if (*env_val == "3.1") {
        return MetalLanguageVersion::METAL_3_1;
      }
      if (*env_val == "3.2") {
        return MetalLanguageVersion::METAL_3_2;
      }
      if (*env_val == "4.0") {
        return MetalLanguageVersion::METAL_4_0;
      }
      TORCH_WARN("Ignoring PYTORCH_MPS_METAL_VERSION=", *env_val, ", expected 3.1, 3.2 or 4.0");
    }
    if (is_macos_at_least(MacOSVersion::MACOS_26_0)) {
      return MetalLanguageVersion::METAL_4_0;
    }
    if (is_macos_at_least(MacOSVersion::MACOS_15_0)) {
      return MetalLanguageVersion::METAL_3_2;
    }
    return MetalLanguageVersion::METAL_3_1;
  }();
  return rc;
}

bool has_mpp() {
#if !defined(CAN_BUILD_METAL_4) && !defined(PYTORCH_JIT_COMPILE_SHADERS)
  // Metal toolchain used to build this binary could not compile the Metal 4.0
  // shaders, so kernels_40.metallib is not embedded into libtorch_cpu
  return false;
#endif
  // MetalPerformancePrimitives needs macOS 26.2+ and shaders compiled with Metal 4.0
  // Also check device family, it's older than M1 on virtualized Mac and it fails to
  // JIT-compile MPP, even though rest of Metal 4 feature work fine
  // https://github.com/pytorch/pytorch/issues/196582
  return is_macos_at_least(MacOSVersion::MACOS_26_2) && metal_language_version() >= MetalLanguageVersion::METAL_4_0 &&
      is_apple_family_or_newer(AppleGPUFamily::APPLE_7_PLUS);
}

} // namespace at::mps
