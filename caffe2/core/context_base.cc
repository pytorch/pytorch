#include "context_base.h"

namespace caffe2 {

// TODO: rename context.h -> context_cpu.h & context_base.h -> context.h
std::array<BaseStaticContext*, COMPILE_TIME_MAX_DEVICE_TYPES>&
GetStaticContexts() {
  static std::array<BaseStaticContext*, COMPILE_TIME_MAX_DEVICE_TYPES>
      static_contexts;
  return static_contexts;
}

void set_static_context(int d, BaseStaticContext* ptr) {
  auto& static_contexts = GetStaticContexts();
  static_contexts[d] = ptr;
}

BaseStaticContext* get_static_context(int d) {
  return GetStaticContexts()[d];
}

} // namespace caffe2
