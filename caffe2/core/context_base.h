#pragma once

#include <ATen/core/context_base.h>
#include "caffe2/core/event.h"

namespace caffe2 {
using at::BaseContext;
using at::BaseStaticContext;
// using at::DeleterFnPtr;

using StaticContextMap = CaffeMap<DeviceType, BaseStaticContext*>;
CAFFE2_API StaticContextMap& GetStaticContexts();
CAFFE2_API void set_static_context(DeviceType t, BaseStaticContext* ptr);
CAFFE2_API BaseStaticContext* get_static_context(DeviceType t);

template <DeviceType t>
struct StaticContextFunctionRegisterer {
  explicit StaticContextFunctionRegisterer(BaseStaticContext* ptr) {
    set_static_context(t, ptr);
  }
};

#define REGISTER_STATIC_CONTEXT(t, f)                                \
  namespace {                                                        \
  static StaticContextFunctionRegisterer<t> g_static_context_##d(f); \
  }

} // namespace caffe2
