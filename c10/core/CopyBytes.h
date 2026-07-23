#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <cstddef>

namespace c10 {

using CopyBytesFunction = void (*)(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device);

struct C10_API _CopyBytesFunctionRegisterer {
  _CopyBytesFunctionRegisterer(
      DeviceType from,
      DeviceType to,
      CopyBytesFunction func_sync,
      CopyBytesFunction func_async = nullptr);
};

#define REGISTER_COPY_BYTES_FUNCTION(from, to, ...)           \
  namespace {                                                 \
  static _CopyBytesFunctionRegisterer C10_ANONYMOUS_VARIABLE( \
      g_copy_function)(from, to, __VA_ARGS__);                \
  }

/*
 * WARNING: Implementations for this function are currently registered from
 * ATen and caffe2, not yet from c10. Don't use this if not either ATen
 * or caffe2 is present as well.
 * We can't move them yet, because the CUDA implementations aren't unified yet
 * between ATen and caffe2.
 * We're planning to move the implementations into c10/backend/xxx
 * to make c10 self contained again.
 */
C10_API void CopyBytes(
    size_t nbytes,
    const void* src,
    Device src_device,
    void* dst,
    Device dst_device,
    bool async);
} // namespace c10
