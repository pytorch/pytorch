#include <ATen/core/context_base.h>

namespace at {

C10_DEFINE_TYPED_REGISTRY(
    ContextRegistry,
    at::DeviceType,
    at::BaseContext,
    std::unique_ptr,
    at::Device);

} // namespace at

namespace caffe2 {

// TODO: rename context.h -> context_cpu.h & context_base.h -> context.h
StaticContextMap& GetStaticContexts() {
  static StaticContextMap static_contexts;
  return static_contexts;
}

void set_static_context(at::DeviceType t, BaseStaticContext* ptr) {
  auto& static_contexts = GetStaticContexts();
  static_contexts[t] = ptr;
}

BaseStaticContext* get_static_context(at::DeviceType t) {
  auto* ptr = GetStaticContexts()[t];
  AT_ASSERTM(ptr, "StaticContext for ", t, " is not registered yet.");
  return ptr;
}

} // namespace caffe2
