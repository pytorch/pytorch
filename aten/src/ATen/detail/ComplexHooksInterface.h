#pragma once

#include <ATen/Registry.h>
#include <ATen/Error.h>

namespace at {

class Context;

struct CAFFE2_API ComplexHooksInterface {
  virtual ~ComplexHooksInterface() {}

  virtual void registerComplexTypes(Context*) const {
    AT_ERROR("Cannot register complex types without loading a library with complex support");
  }
};

struct CAFFE2_API ComplexHooksArgs {};

namespace detail {
CAFFE2_API const ComplexHooksInterface& getComplexHooks();
}

}

C10_DECLARE_REGISTRY(ComplexHooksRegistry, at::ComplexHooksInterface, at::ComplexHooksArgs)
#define REGISTER_COMPLEX_HOOKS(clsname) \
  C10_REGISTER_CLASS(ComplexHooksRegistry, clsname, clsname)
