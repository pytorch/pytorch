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

namespace at {
void BaseStaticContext::ExtractDeviceOption(
    caffe2::DeviceOption* device,
    const void* /*data*/) {
  device->set_device_type(caffe2::TypeToProto(GetDeviceType()));
}

inline void BaseContext::CopyItemsSameDevice(
    const caffe2::TypeMeta& meta,
    size_t n,
    const void* src,
    void* dst) {
  if (meta.copy()) {
    EnforceMetaCopyOK();
    meta.copy()(src, dst, n);
  } else {
    CopyBytesSameDevice(n * meta.itemsize(), src, dst);
  }
}

inline void BaseContext::CopyItemsFromCPU(
    const caffe2::TypeMeta& meta,
    size_t n,
    const void* src,
    void* dst) {
  if (meta.copy()) {
    EnforceMetaCopyOK();
    meta.copy()(src, dst, n);
  } else {
    CopyBytesFromCPU(n * meta.itemsize(), src, dst);
  }
}

inline void BaseContext::CopyItemsToCPU(
    const caffe2::TypeMeta& meta,
    size_t n,
    const void* src,
    void* dst) {
  if (meta.copy()) {
    EnforceMetaCopyOK();
    meta.copy()(src, dst, n);
  } else {
    CopyBytesToCPU(n * meta.itemsize(), src, dst);
  }
}

} // namespace at
