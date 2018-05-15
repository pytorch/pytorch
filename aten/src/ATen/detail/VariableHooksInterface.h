#pragma once

#include <ATen/Registry.h>
#include <ATen/Error.h>
#include <ATen/ScalarType.h>

namespace at {
  class Context;
}

// NB: Registry class not actually in the namespace detail, due to limitations
// of Registry.h
namespace at {

// The VariableHooksInterface is an interface for autograd functionality
// which currently doesn't live in libATen.so AND needs to be called from
// ATen.  In this case, it is only the type registry for Variable types,
// letting us add extra variables types if CUDA types are initialized lazily.
//
// We may choose to absorb autograd into ATen, in which case this interface is obsolete.
//
struct AT_API VariableHooksInterface {

  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~VariableHooksInterface() {}

  virtual void registerVariableTypeFor(Context*, Backend backend, ScalarType scalar_type) const {
    // no-op if Variable not available; it'll get handled (if at all) when
    // libtorch.so gets loaded
  }

};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
struct AT_API VariableHooksArgs {};

AT_DECLARE_REGISTRY(VariableHooksRegistry, VariableHooksInterface, VariableHooksArgs)
#define REGISTER_VARIABLE_HOOKS(clsname) AT_REGISTER_CLASS(VariableHooksRegistry, clsname, clsname)

namespace detail {
  AT_API const VariableHooksInterface& getVariableHooks();
}

} // namespace at
