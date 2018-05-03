#pragma once

#include <ATen/Registry.h>
#include <ATen/Error.h>
#include <ATen/ScalarType.h>

namespace at {
  class Context;
}

namespace at { namespace detail {

// The VariableHooksInterface is an interface for autograd functionality
// which currently doesn't live in libATen.so AND needs to be called from
// ATen.  In this case, it is only the type registry for Variable types,
// letting us add extra variables types if CUDA types are initialized lazily.
//
// We may choose to absorb autograd into ATen, in which case this interface is obsolete.
//
struct VariableHooksInterface {

  virtual void registerVariableTypeFor(Context*, Backend backend, ScalarType scalar_type) const {
    // no-op if Variable not available; it'll get handled (if at all) when
    // libtorch.so gets loaded
  }

};

AT_DECLARE_REGISTRY(VariableHooksRegistry, VariableHooksInterface);
#define REGISTER_VARIABLE_HOOKS(clsname) AT_REGISTER_CLASS(VariableHooksRegistry, clsname, clsname)

const VariableHooksInterface& getVariableHooks();

}} // namespace at::detail
