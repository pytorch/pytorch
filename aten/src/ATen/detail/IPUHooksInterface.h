#pragma once

#include <ATen/core/Generator.h>
#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

namespace at {

struct TORCH_API IPUHooksInterface {
  virtual ~IPUHooksInterface() = default;

  virtual const Generator& getDefaultIPUGenerator(
      DeviceIndex device_index = -1) const {
    AT_ERROR(
        "Cannot get the default IPU generator: the IPU backend is not "
        "available.");
  }

  virtual const Generator& newIPUGenerator(
      DeviceIndex device_index = -1) const {
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
