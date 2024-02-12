#pragma once

#include <c10/util/Exception.h>

#include <c10/util/Registry.h>

#include <ATen/detail/AcceleratorHooksInterface.h>

#include <string>

namespace at {
class Context;
}

namespace at {

constexpr const char* MTIA_HELP =
    "The MTIA backend requires MTIA extension for PyTorch;"
    "this error has occurred because you are trying "
    "to use some MTIA's functionality without MTIA extension included.";

struct TORCH_API MTIAHooksInterface : AcceleratorHooksInterface {
  virtual ~MTIAHooksInterface() override = default;

  virtual void initMTIA() const {
    TORCH_CHECK(
        false,
        "Cannot initialize MTIA without MTIA Extension for PyTorch.",
        MTIA_HELP);
  }

  virtual bool hasMTIA() const {
    return false;
  }

  virtual std::string showConfig() const {
    TORCH_CHECK(
        false,
        "Cannot query detailed MTIA version without MTIA Extension for PyTorch.",
        MTIA_HELP);
  }

  virtual bool hasPrimaryContext(DeviceIndex device_index) const override {
    TORCH_CHECK(
        false,
        "Cannot check MTIA primary context without MTIA Extension for PyTorch.",
        MTIA_HELP);
  }

};

struct TORCH_API MTIAHooksArgs {};

C10_DECLARE_REGISTRY(MTIAHooksRegistry, MTIAHooksInterface, MTIAHooksArgs);
#define REGISTER_MTIA_HOOKS(clsname) \
  C10_REGISTER_CLASS(MTIAHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const MTIAHooksInterface& getMTIAHooks();
} // namespace detail
} // namespace at
