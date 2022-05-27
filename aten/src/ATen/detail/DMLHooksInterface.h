#pragma once

#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

constexpr const char* DML_HELP =
  " You need to 'import torch_dml' to use the 'dml' device in PyTorch. "
  "The 'torch_dml' module is provided by "
  "(https://github.com/microsoft/directml).";

// NB: Class must live in `at` due to limitations of Registry.h.
namespace at {

struct TORCH_API DMLHooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~DMLHooksInterface() {}

  virtual std::string showConfig() const {
    TORCH_CHECK(false, "Cannot query detailed DML version information.", DML_HELP);
  }

  virtual bool hasDML() const {
    return false;
  }
};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
struct TORCH_API DMLHooksArgs {};

C10_DECLARE_REGISTRY(DMLHooksRegistry, DMLHooksInterface, DMLHooksArgs);
#define REGISTER_DML_HOOKS(clsname) \
  C10_REGISTER_CLASS(DMLHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const DMLHooksInterface& getDMLHooks();
} // namespace detail

} // namespace at
