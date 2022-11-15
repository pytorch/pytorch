//  Copyright © 2022 Apple Inc.

#pragma once
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <ATen/ATen.h>


#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
typedef id<MTLDevice> MTLDevice_t;
typedef id<MTLLibrary> MTLLibrary_t;
typedef id<MTLFunction> MTLFunction_t;
typedef MTLFunctionConstantValues* MTLFunctionConstantValues_t;
#else
typedef void* MTLDevice;
typedef void* MTLDevice_t;
typedef void* MTLLibrary_t;
typedef void* MTLFunction_t;
typedef void* MTLFunctionConstantValues_t;
#endif

using namespace std;

namespace at {
namespace mps {

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

  MTLFunction_t metalIndexingFunction(const std::string &kernel, MTLFunctionConstantValues_t constantValues);

  ~MPSDevice();

 private:
  static MPSDevice* _device;
  MTLDevice_t _mtl_device;
  MTLLibrary_t _mtl_indexing_library;
  MPSDevice();
};

TORCH_API bool is_available();

TORCH_API at::Allocator* GetMPSAllocator(bool useSharedAllocator = false);

} // namespace mps
} // namespace at
