//  Copyright © 2022 Apple Inc.

#pragma once
#include <ATen/Device.h>
#include <c10/core/Allocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#ifdef __OBJC__
// Metal.h pulls in Foundation.h, which transitively includes CarbonCore
// headers that emit a flood of -Wdeprecated-declarations on recent macOS SDKs.
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
#include <Metal/Metal.h>
C10_DIAGNOSTIC_POP()
typedef id<MTLDevice> MTLDevice_t;
#else
typedef void* MTLDevice_t;
#endif

namespace at::mps {

// Helper enum to check if a MPSGraph op is supported in a given macOS version
enum class MacOSVersion : uint32_t {
  MACOS_14_4 = 0,
  MACOS_15_0,
  MACOS_15_1,
  MACOS_15_2,
  MACOS_26_0,
  MACOS_26_2,
  MACOS_26_4,
  MACOS_27_0,
};

// Metal language version to compile shaders with
// Values match MTLLanguageVersion
enum class MetalLanguageVersion : uint32_t {
  METAL_3_1 = (3 << 16) + 1,
  METAL_3_2 = (3 << 16) + 2, // allows lambdas in shader code
  METAL_4_0 = (4 << 16) + 0, // allows tensor template arguments
};

// Helper enum for GPU-family-gated workarounds
enum class AppleGPUFamily : uint32_t {
  APPLE_7_PLUS = 1007, // M1
  APPLE_8_PLUS = 1008, // M2
  APPLE_9_PLUS = 1009, // M3 / M4
  APPLE_10_PLUS = 1010, // M5
};

//-----------------------------------------------------------------
//  MPSDevice
//
// MPSDevice is a singleton class that returns the default device
//-----------------------------------------------------------------

class TORCH_API MPSDevice {
 public:
  /**
   * MPSDevice should not be cloneable.
   */
  MPSDevice(MPSDevice& other) = delete;
  /**
   * MPSDevice should not be assignable.
   */
  void operator=(const MPSDevice&) = delete;
  /**
   * Gets single instance of the Device.
   */
  static MPSDevice* getInstance();
  /**
   * Returns the single device.
   */
  MTLDevice_t device() {
    return _mtl_device;
  }
  /**
   * Returns whether running on Ventura or newer
   */
  bool isMacOS13Plus(MacOSVersion version) const;

  /**
   * Returns device name
   */
  std::string getName() const;

  /**
   * Returns number of GPU cores.
   * 1 Core = 16 ExecutionUnit x 8 ALU x 24 threads
   */
  unsigned getCoreCount() const;

  ~MPSDevice();

 private:
  static MPSDevice* _device;
  MTLDevice_t _mtl_device;
  MPSDevice();
};

TORCH_API bool is_available();
TORCH_API bool is_macos_at_least(MacOSVersion version);
TORCH_API bool is_apple_family_or_newer(AppleGPUFamily family);
// Metal language version to compile shaders with. Derived from the OS version,
// but PYTORCH_MPS_METAL_VERSION overrides it outright, so requesting a version
// the host cannot provide fails rather than silently falling back
TORCH_API MetalLanguageVersion metal_language_version();
// Whether MetalPerformancePrimitives (cooperative tensors) is usable, i.e. the
// OS is new enough, the Metal 4.0 shaders were compiled into this binary and
// PYTORCH_MPS_METAL_VERSION did not ask for an older language version
TORCH_API bool has_mpp();
TORCH_API at::Allocator* GetMPSAllocator();

inline Device getDeviceFromPtr(void* ptr) {
  return {c10::DeviceType::MPS, 0};
}

} // namespace at::mps
