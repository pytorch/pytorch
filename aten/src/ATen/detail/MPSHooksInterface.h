//  Copyright © 2022 Apple Inc.

#pragma once

#include <c10/core/Allocator.h>
#include <ATen/core/Generator.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>

namespace at {
class Context;
}

namespace at {

struct TORCH_API MPSHooksInterface {
  virtual ~MPSHooksInterface() = default;

  // Initialize the MPS library state
  virtual void initMPS() const {
    AT_ERROR("Cannot initialize MPS without MPS backend.");
  }

  virtual bool hasMPS() const {
    return false;
  }

  virtual bool isOnMacOS13orNewer() const {
    AT_ERROR("MPS backend is not available.");
  }

  virtual const Generator& getDefaultMPSGenerator() const {
    AT_ERROR("Cannot get default MPS generator without MPS backend.");
  }

  virtual Allocator* getMPSDeviceAllocator() const {
    AT_ERROR("MPSDeviceAllocator requires MPS.");
  }

  virtual void deviceSynchronize() const {
    AT_ERROR("Cannot synchronize MPS device without MPS backend.");
  }

  virtual void emptyCache() const {
    AT_ERROR("Cannot execute emptyCache() without MPS backend.");
  }

  virtual size_t getCurrentAllocatedMemory() const {
    AT_ERROR("Cannot execute getCurrentAllocatedMemory() without MPS backend.");
  }

  virtual size_t getDriverAllocatedMemory() const {
    AT_ERROR("Cannot execute getDriverAllocatedMemory() without MPS backend.");
  }

  virtual void setMemoryFraction(double /*ratio*/) const {
    AT_ERROR("Cannot execute setMemoryFraction() without MPS backend.");
  }
};

struct TORCH_API MPSHooksArgs {};

C10_DECLARE_REGISTRY(MPSHooksRegistry, MPSHooksInterface, MPSHooksArgs);
#define REGISTER_MPS_HOOKS(clsname) \
  C10_REGISTER_CLASS(MPSHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const MPSHooksInterface& getMPSHooks();

} // namespace detail
} // namespace at
