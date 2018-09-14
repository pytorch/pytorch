#pragma once

#include <ATen/core/context_base.h>
// For CaffeMap
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {
using at::BaseContext;
using at::BaseStaticContext;

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

// Context constructor registry
CAFFE_DECLARE_TYPED_REGISTRY(
    ContextRegistry,
    DeviceType,
    BaseContext,
    std::unique_ptr,
    caffe2::DeviceOption);

#define REGISTER_CONTEXT(type, ...) \
  CAFFE_REGISTER_TYPED_CLASS(ContextRegistry, type, __VA_ARGS__)

inline std::unique_ptr<BaseContext> CreateContext(
    DeviceType type,
    const caffe2::DeviceOption& option = caffe2::DeviceOption()) {
  return ContextRegistry()->Create(type, option);
}

} // namespace caffe2
