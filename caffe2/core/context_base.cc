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

} // namespace caffe2
