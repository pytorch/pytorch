#pragma once

#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

namespace at {

class Context;

struct CAFFE2_API QIntHooksInterface {
  virtual ~QIntHooksInterface() {}

  virtual void registerQIntTypes(Context*) const {
    AT_ERROR("Cannot register qint types without loading a library with qint support");
  }
};

struct CAFFE2_API QIntHooksArgs {};
C10_DECLARE_REGISTRY(
    QIntHooksRegistry,
    QIntHooksInterface,
    QIntHooksArgs);
#define REGISTER_QINT_HOOKS(clsname)                            \
  C10_REGISTER_CLASS(QIntHooksRegistry, clsname, clsname)

namespace detail {
  CAFFE2_API const QIntHooksInterface& getQIntHooks();
}

}
