#pragma once

#include <c10/macros/Export.h>

namespace at {

struct CAFFE2_API XLAHooksInterface {
  virtual ~XLAHooksInterface() {}

  virtual bool hasXLA() const {
    return false;
  }

  virtual int getNumDevices() const {
    return 0;
  }
};

namespace detail {

CAFFE2_API const XLAHooksInterface& getXLAHooks();
CAFFE2_API void setXLAHooks(XLAHooksInterface* xla_hooks);

} // namespace detail
} // namespace at
