#pragma once

#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <ATen/detail/AcceleratorHooksInterface.h>

// NB: Class must live in `at` due to limitations of Registry.h.
namespace at {

struct TORCH_API MAIAHooksInterface : AcceleratorHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  ~MAIAHooksInterface() override = default;

  void init() const override {
    TORCH_CHECK(false, "Cannot initialize MAIA without ATen_maia library.");
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    TORCH_CHECK(false, "Cannot initialize MAIA without ATen_maia library.");
    return false;
  }

  std::string showConfig() const override {
    TORCH_CHECK(false, "Cannot query detailed MAIA version information.");
  }
};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
struct TORCH_API MAIAHooksArgs {};

TORCH_DECLARE_REGISTRY(MAIAHooksRegistry, MAIAHooksInterface, MAIAHooksArgs);
#define REGISTER_MAIA_HOOKS(clsname) \
  C10_REGISTER_CLASS(MAIAHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const MAIAHooksInterface& getMAIAHooks();
} // namespace detail

} // namespace at
