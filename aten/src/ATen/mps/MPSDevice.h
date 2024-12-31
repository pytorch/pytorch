//  Copyright © 2022 Apple Inc.

#pragma once
#include <c10/core/Allocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
typedef id<MTLDevice> MTLDevice_t;
#else
typedef void* MTLDevice_t;
#endif

namespace at::mps {

// Helper enum to check if a MPSGraph op is supported in a given macOS version
enum class MacOSVersion : uint32_t {
  MACOS_VER_13_1_PLUS = 0,
  MACOS_VER_13_2_PLUS,
  MACOS_VER_13_3_PLUS,
  MACOS_VER_14_0_PLUS,
  MACOS_VER_14_4_PLUS,
  MACOS_VER_15_0_PLUS,
  MACOS_VER_15_1_PLUS,
  MACOS_VER_15_2_PLUS,
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

  ~MPSDevice();

 private:
  static MPSDevice* _device;
  MTLDevice_t _mtl_device;
  MPSDevice();
};

TORCH_API bool is_available();
TORCH_API bool is_macos_13_or_newer(MacOSVersion version);
TORCH_API at::Allocator* GetMPSAllocator(bool useSharedAllocator = false);

} // namespace at::mps
