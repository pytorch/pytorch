#pragma once

#include <ATen/core/context_base.h>
// For CaffeMap
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {
using at::BaseContext;
using at::BaseStaticContext;

#define CAFFE_DECLARE_POINTER_REGISTRY(RegistryName, SrcType, ObjectType, ...) \
  CAFFE2_EXPORT Registry<SrcType, ObjectType, ##__VA_ARGS__>* RegistryName();  \
  typedef Registerer<SrcType, ObjectType, ##__VA_ARGS__>                       \
      Registerer##RegistryName;

#define CAFFE_DEFINE_POINTER_REGISTRY(RegistryName, SrcType, ObjectType, ...)  \
  CAFFE2_EXPORT Registry<SrcType, ObjectType, ##__VA_ARGS__>* RegistryName() { \
    static Registry<SrcType, ObjectType, ##__VA_ARGS__>* registry =            \
        new Registry<SrcType, ObjectType, ##__VA_ARGS__>();                    \
    return registry;                                                           \
  }

#define CAFFE_REGISTER_POINTER(RegistryName, key, ...)      \
  namespace {                                               \
  static Registerer##RegistryName CAFFE_ANONYMOUS_VARIABLE( \
      g_##RegistryName)(key, RegistryName(), __VA_ARGS__);  \
  }

using StaticContextMap = CaffeMap<DeviceType, BaseStaticContext*>;
CAFFE2_API StaticContextMap& GetStaticContexts();
CAFFE2_API void set_static_context(DeviceType t, BaseStaticContext* ptr);
CAFFE2_API BaseStaticContext* get_static_context(DeviceType t);

template <DeviceType t>
struct StaticContextRegisterer {
  explicit StaticContextRegisterer(BaseStaticContext* ptr) {
    set_static_context(t, ptr);
  }
};

#define REGISTER_STATIC_CONTEXT(t, f)                        \
  namespace {                                                \
  static StaticContextRegisterer<t> g_static_context_##d(f); \
  }

// ExtractDeviceOption registry
// typedef void (*ExtractDeviceOptionFnPtr)(DeviceOption*, const void*);
struct ExtractDeviceOptionFn {
  virtual ~ExtractDeviceOptionFn() {}
  virtual void operator()(DeviceOption*, const void*) = 0;
};

CAFFE_DECLARE_POINTER_REGISTRY(
    ExtractDeviceOptionFnRegistry,
    DeviceType,
    ExtractDeviceOptionFn*);

#define REGISTER_DEVICE_OPTION_FN(t, ...) \
  CAFFE_REGISTER_POINTER(ExtractDeviceOptionFnRegistry, t, __VA_ARGS__)

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

inline void ExtractDeviceOption(
    DeviceOption* device,
    const void* data,
    DeviceType device_type) {
  CAFFE_ENFORCE(data, "data cannot be nullptr");
  (*ExtractDeviceOptionFnRegistry()->Create(device_type))(device, data);
}

} // namespace caffe2
