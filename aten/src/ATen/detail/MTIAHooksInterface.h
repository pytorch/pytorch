#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace at {
class Context;
}

// We use forward declaration here instead of #include <ATen/dlpack.h> to avoid
// leaking DLPack implementation detail to every project that includes `ATen/Context.h`, which in turn
// would lead to a conflict when linked with another project using DLPack (for example TVM)
struct DLDevice_;

namespace at {

constexpr const char* MTIA_HELP =
    "The MTIA backend requires MTIA extension for PyTorch;"
    "this error has occurred because you are trying "
    "to use some MTIA's functionality without MTIA extension included.";

struct TORCH_API MTIAHooksInterface {
  virtual ~MTIAHooksInterface() = default;

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
};

struct TORCH_API MTIAHooksArgs {};

C10_DECLARE_REGISTRY(MTIAHooksRegistry, MTIAHooksInterface, MTIAHooksArgs);
#define REGISTER_MTIA_HOOKS(clsname) \
  C10_REGISTER_CLASS(MTIAHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const MTIAHooksInterface& getMTIAHooks();
} // namespace detail
} // namespace at
