#pragma once

#include <ATen/Registry.h>
#include <ATen/Error.h>

namespace at {

struct Context;

struct AT_API ComplexHooksInterface {
  virtual ~ComplexHooksInterface() {}

  virtual void registerComplexTypes(Context*) const {
    AT_ERROR("Cannot register complex types without loading a library with complex support");
  }
};

struct AT_API ComplexHooksArgs {};
AT_DECLARE_REGISTRY(ComplexHooksRegistry, ComplexHooksInterface, ComplexHooksArgs)
#define REGISTER_COMPLEX_HOOKS(clsname) \
  AT_REGISTER_CLASS(ComplexHooksRegistry, clsname, clsname)

namespace detail {
AT_API const ComplexHooksInterface& getComplexHooks();
}

}
