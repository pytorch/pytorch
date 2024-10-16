#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/util/Exception.h>

#include <c10/util/Registry.h>

#include <ATen/detail/AcceleratorHooksInterface.h>

#include <memory>

namespace at {
class Context;
}

// NB: Class must live in `at` due to limitations of Registry.h.
namespace at {

// The HIPHooksInterface is an omnibus interface for any HIP functionality
// which we may want to call into from CPU code (and thus must be dynamically
// dispatched, to allow for separate compilation of HIP code).  See
// CUDAHooksInterface for more detailed motivation.
struct TORCH_API HIPHooksInterface : AcceleratorHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  ~HIPHooksInterface() override = default;

  // Initialize the HIP library state
  virtual void initHIP() const {
    AT_ERROR("Cannot initialize HIP without ATen_hip library.");
  }

  virtual std::unique_ptr<c10::GeneratorImpl> initHIPGenerator(Context*) const {
    AT_ERROR("Cannot initialize HIP generator without ATen_hip library.");
  }

  virtual bool hasHIP() const {
    return false;
  }

  virtual c10::DeviceIndex current_device() const {
    return -1;
  }

  bool isPinnedPtr(const void* data) const override {
    return false;
  }

  Allocator* getPinnedMemoryAllocator() const override {
    AT_ERROR("Pinned memory requires HIP.");
  }

  virtual void registerHIPTypes(Context*) const {
    AT_ERROR("Cannot registerHIPTypes() without ATen_hip library.");
  }

  virtual int getNumGPUs() const {
    return 0;
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    AT_ERROR("Cannot check primary context without ATen_hip library.");
  }
};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
struct TORCH_API HIPHooksArgs {};

TORCH_DECLARE_REGISTRY(HIPHooksRegistry, HIPHooksInterface, HIPHooksArgs);
#define REGISTER_HIP_HOOKS(clsname) \
  C10_REGISTER_CLASS(HIPHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const HIPHooksInterface& getHIPHooks();

} // namespace detail
} // namespace at
