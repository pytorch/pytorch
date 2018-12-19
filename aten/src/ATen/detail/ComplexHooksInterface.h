#pragma once

#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

namespace at {

class Context;

struct CAFFE2_API ComplexHooksInterface {
  virtual ~ComplexHooksInterface() {}

  virtual void registerComplexTypes(Context*) const {
    AT_ERROR("Cannot register complex types without loading a library with complex support");
  }
};

struct CAFFE2_API ComplexHooksArgs {};
C10_DECLARE_REGISTRY(
    ComplexHooksRegistry,
    ComplexHooksInterface,
    ComplexHooksArgs);
#define REGISTER_COMPLEX_HOOKS(clsname) \
  C10_REGISTER_CLASS(ComplexHooksRegistry, clsname, clsname)

namespace detail {
CAFFE2_API const ComplexHooksInterface& getComplexHooks();
}

}
