#pragma once

#include <ATen/core/Generator.h>
#include <ATen/detail/AcceleratorHooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

namespace at {

struct TORCH_API IPUHooksInterface: AcceleratorHooksInterface {
  ~IPUHooksInterface() override = default;

  void init() const override {
    TORCH_CHECK(false, "Cannot initialize IPU without ATen_ipu library.");
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    TORCH_CHECK(false, "Cannot initialize IPU without ATen_ipu library.");
    return false;
  }

  virtual const Generator& getDefaultIPUGenerator(
      DeviceIndex device_index [[maybe_unused]] = -1) const {
    AT_ERROR(
        "Cannot get the default IPU generator: the IPU backend is not "
        "available.");
  }

  virtual Generator newIPUGenerator(DeviceIndex device_index [[maybe_unused]] = -1) const {
    AT_ERROR(
        "Cannot create a new IPU generator: the IPU backend is not available.");
  }
};

struct TORCH_API IPUHooksArgs {};

TORCH_DECLARE_REGISTRY(IPUHooksRegistry, IPUHooksInterface, IPUHooksArgs);
#define REGISTER_IPU_HOOKS(clsname) \
  C10_REGISTER_CLASS(IPUHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const IPUHooksInterface& getIPUHooks();
} // namespace detail
} // namespace at
