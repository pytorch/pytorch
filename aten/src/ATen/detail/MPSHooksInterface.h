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
  // this fails the implementation if MPSHooks functions are called, but
  // MPS backend is not present.
  #define FAIL_MPSHOOKS_FUNC(func) \
    TORCH_CHECK(false, "Cannot execute ", func, "() without MPS backend.");

  virtual ~MPSHooksInterface() = default;

  // Initialize the MPS library state
  virtual void initMPS() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual bool hasMPS() const {
    return false;
  }
  virtual bool isOnMacOS13orNewer(unsigned minor = 0) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual const Generator& getDefaultMPSGenerator() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual Allocator* getMPSDeviceAllocator() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void deviceSynchronize() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void commitStream() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void* getCommandBuffer() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void* getDispatchQueue() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void emptyCache() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual size_t getCurrentAllocatedMemory() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual size_t getDriverAllocatedMemory() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void setMemoryFraction(double /*ratio*/) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void profilerStartTrace(const std::string& mode, bool waitUntilCompleted) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void profilerStopTrace() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual uint32_t acquireEvent(bool enable_timing) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void releaseEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void recordEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void waitForEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void synchronizeEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual bool queryEvent(uint32_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual double elapsedTimeOfEvents(uint32_t start_event_id, uint32_t end_event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }

  #undef FAIL_MPSHOOKS_FUNC
};

struct TORCH_API MPSHooksArgs {};

TORCH_DECLARE_REGISTRY(MPSHooksRegistry, MPSHooksInterface, MPSHooksArgs);
#define REGISTER_MPS_HOOKS(clsname) \
  C10_REGISTER_CLASS(MPSHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const MPSHooksInterface& getMPSHooks();

} // namespace detail
} // namespace at
