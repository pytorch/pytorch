#pragma once

#include <cstdlib>
#include <ctime>
#include <memory>
#include <unordered_map>

#include "caffe2/core/allocator.h"
#include "caffe2/core/event.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/typeid.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

class BaseContext;

/* BaseStaticContext defines the interface for static context, which contains
   functions that are invoked statically before in Tensor class, e.g. New,
   We will merge this with Allocator later.
 */
class CAFFE2_API BaseStaticContext {
 public:
  virtual ~BaseStaticContext() noexcept {}

  virtual std::pair<void*, MemoryDeleter> New(size_t nbytes) const = 0;

  virtual std::unique_ptr<BaseContext> CreateContext() = 0;

  virtual std::unique_ptr<BaseContext> CreateContext(const DeviceOption&) = 0;

  virtual DeviceType GetDeviceType() = 0;

  /*
   * @brief: Sets the DeviceOption for argument `device` based on the
   * current context and the a data pointer
   */
  virtual void ExtractDeviceOption(DeviceOption* device, const void* /*data*/) {
    device->set_device_type(GetDeviceType());
  }
};

/**
 * Virtual interface for the Context class in Caffe2.
 *
 * A Context defines all the necessities to run an operator on a specific
 * device. Specific Context classes needs to implement all the pure virtual
 * functions in the BaseContext class.
 * TODO: add docs after this is finalized.
 */
class CAFFE2_API BaseContext {
 public:
  virtual ~BaseContext() noexcept {}

  virtual BaseStaticContext* GetStaticContext() const = 0;

  /* Sorry for the naming, will get rid of this in future diff */
  virtual DeviceType GetDevicetype() const = 0;

  virtual void SwitchToDevice(int /*stream_id*/) = 0;

  inline void SwitchToDevice() {
    SwitchToDevice(0);
  }

  virtual void WaitEvent(const Event& ev) = 0;

  virtual void Record(Event* ev, const char* err_msg = nullptr) const = 0;

  virtual void FinishDeviceComputation() = 0;

  // This used to be arbitrary cross-device copy, but it turns out everyone
  // did direct CPU-X copy, so we just make three functions for it (to avoid
  // double dispatch).  This will get obsoleted by C10. where copies
  // will be proper operators (and get to rely on multiple dispatch there.)
  virtual void
  CopyBytesSameDevice(size_t nbytes, const void* src, void* dst) = 0;

  virtual void CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) = 0;

  virtual void CopyBytesToCPU(size_t nbytes, const void* src, void* dst) = 0;

  virtual void CopyBytesToDevice(
      size_t nbytes,
      const void* src,
      void* dst,
      DeviceType type) {
    if (type == CPU) {
      CopyBytesToCPU(nbytes, src, dst);
    } else if (type == GetDevicetype()) {
      CopyBytesSameDevice(nbytes, src, dst);
    } else {
      CAFFE_THROW(
          "CopyBytesToDevice can only copy to CPU or between same "
          "device. Can't copy from: ",
          GetDevicetype(),
          " to",
          type);
    }
  }

  template <typename T>
  inline void CopySameDevice(size_t n, const T* src, T* dst) {
    static_assert(
        std::is_fundamental<T>::value,
        "CopySameDevice requires fundamental types");
    CopyBytesSameDevice(
        n * sizeof(T), static_cast<const void*>(src), static_cast<void*>(dst));
  }

  template <typename T>
  inline void CopyFromCPU(size_t n, const T* src, T* dst) {
    static_assert(
        std::is_fundamental<T>::value,
        "CopyFromCPU requires fundamental types");
    CopyBytesFromCPU(
        n * sizeof(T), static_cast<const void*>(src), static_cast<void*>(dst));
  }

  template <typename T>
  inline void CopyToCPU(size_t n, const T* src, T* dst) {
    static_assert(
        std::is_fundamental<T>::value, "CopyToCPU requires fundamental types");
    CopyBytesToCPU(
        n * sizeof(T), static_cast<const void*>(src), static_cast<void*>(dst));
  }

  virtual bool SupportsNonFundamentalTypes() const {
    return false;
  }

  inline void EnforceMetaCopyOK() {
    CAFFE_ENFORCE(
        SupportsNonFundamentalTypes(), "Context requires fundamental types");
  }

  inline void CopyItemsSameDevice(
      const TypeMeta& meta,
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

  inline void
  CopyItemsFromCPU(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    if (meta.copy()) {
      EnforceMetaCopyOK();
      meta.copy()(src, dst, n);
    } else {
      CopyBytesFromCPU(n * meta.itemsize(), src, dst);
    }
  }

  inline void
  CopyItemsToCPU(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    if (meta.copy()) {
      EnforceMetaCopyOK();
      meta.copy()(src, dst, n);
    } else {
      CopyBytesToCPU(n * meta.itemsize(), src, dst);
    }
  }

  static BaseStaticContext* static_context_[COMPILE_TIME_MAX_DEVICE_TYPES];

  template <int d>
  friend struct StaticContextFunctionRegisterer;
};

template <int d>
struct StaticContextFunctionRegisterer {
  explicit StaticContextFunctionRegisterer(BaseStaticContext* ptr) {
    static_assert(d < COMPILE_TIME_MAX_DEVICE_TYPES, "");
    BaseContext::static_context_[d] = ptr;
  }
};

#define REGISTER_STATIC_CONTEXT(d, f)                                \
  namespace {                                                        \
  static StaticContextFunctionRegisterer<d> g_static_context_##d(f); \
  }

#define GET_STATIC_CONTEXT(d) BaseContext::static_context_[d]
} // namespace caffe2
