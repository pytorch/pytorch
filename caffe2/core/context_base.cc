#include "context_base.h"

namespace caffe2 {

// TODO: rename context.h -> context_cpu.h & context_base.h -> context.h
StaticContextMap& GetStaticContexts() {
  static StaticContextMap static_contexts;
  return static_contexts;
}

void set_static_context(DeviceType t, BaseStaticContext* ptr) {
  auto& static_contexts = GetStaticContexts();
  static_contexts[t] = ptr;
}

BaseStaticContext* get_static_context(DeviceType t) {
  auto* ptr = GetStaticContexts()[t];
  CAFFE_ENFORCE(ptr, "StaticContext is not registered yet.");
  return ptr;
}

CAFFE_DEFINE_POINTER_REGISTRY(
    ExtractDeviceOptionFnRegistry,
    DeviceType,
    ExtractDeviceOptionFn*);

CAFFE_DEFINE_TYPED_REGISTRY(
    ContextRegistry,
    DeviceType,
    BaseContext,
    std::unique_ptr,
    DeviceOption);

} // namespace caffe2
