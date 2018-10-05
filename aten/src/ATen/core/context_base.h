#pragma once

#include <array>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <unordered_map>

#include <ATen/core/ATenGeneral.h>
#include <ATen/core/Allocator.h>
#include <ATen/core/Device.h>
#include <ATen/core/Error.h>
#include <ATen/core/UniqueVoidPtr.h>
#include <ATen/core/typeid.h>
#include <c10/util/Registry.h>

namespace caffe2 {
class Event;
class DeviceOption;

} // namespace caffe2
namespace at {

class BaseContext;

/* BaseStaticContext defines the interface for static context, which contains
   functions that are invoked statically before in Tensor class, e.g. New,
   We will merge this with Allocator later.
 */
class CAFFE2_API BaseStaticContext {
 public:
  virtual ~BaseStaticContext() noexcept {}

  virtual at::DataPtr New(size_t nbytes) const = 0;

  virtual DeviceType GetDeviceType() = 0;

  /*
   * @brief: Sets the DeviceOption for argument `device` based on the
   * current context and the a data pointer
   */
  virtual void ExtractDeviceOption(
      caffe2::DeviceOption* device,
      const void* /*data*/) = 0;
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
  virtual DeviceType device_type() const = 0;

  virtual void SwitchToDevice(int /*stream_id*/) = 0;

  inline void SwitchToDevice() {
    SwitchToDevice(0);
  }

  virtual void WaitEvent(const caffe2::Event& ev) = 0;

  virtual void Record(caffe2::Event* ev, const char* err_msg = nullptr)
      const = 0;

  virtual void FinishDeviceComputation() = 0;

  // This used to be arbitrary cross-device copy, but it turns out everyone
  // did direct CPU-X copy, so we just make three functions for it (to avoid
  // double dispatch).  This will get obsoleted by C10. where copies
  // will be proper operators (and get to rely on multiple dispatch there.)
  virtual void CopyBytesSameDevice(
      size_t nbytes,
      const void* src,
      void* dst) = 0;

  virtual void CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) = 0;

  virtual void CopyBytesToCPU(size_t nbytes, const void* src, void* dst) = 0;

  virtual void CopyBytesToDevice(
      size_t nbytes,
      const void* src,
      void* dst,
      DeviceType type) {
    if (type == DeviceType::CPU) {
      CopyBytesToCPU(nbytes, src, dst);
    } else if (type == device_type()) {
      CopyBytesSameDevice(nbytes, src, dst);
    } else {
      AT_ERROR(
          "CopyBytesToDevice can only copy to CPU or between same "
          "device. Can't copy from: ",
          device_type(),
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
    AT_ASSERTM(
        SupportsNonFundamentalTypes(), "Context requires fundamental types");
  }

  void CopyItemsSameDevice(
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

  void CopyItemsFromCPU(
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

  void CopyItemsToCPU(
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
};

// Context constructor registry
C10_DECLARE_TYPED_REGISTRY(
    ContextRegistry,
    at::DeviceType,
    at::BaseContext,
    std::unique_ptr,
    at::Device);

#define REGISTER_CONTEXT(type, ...) \
  C10_REGISTER_TYPED_CLASS(ContextRegistry, type, __VA_ARGS__)

inline std::unique_ptr<at::BaseContext> CreateContext(
    const at::Device& device) {
  return at::ContextRegistry()->Create(device.type(), device);
}

} // namespace at

namespace caffe2 {

using at::BaseContext;
using at::BaseStaticContext;

using StaticContextMap = std::unordered_map<at::DeviceType, BaseStaticContext*>;
CAFFE2_API StaticContextMap& GetStaticContexts();
CAFFE2_API void set_static_context(at::DeviceType t, BaseStaticContext* ptr);
CAFFE2_API BaseStaticContext* get_static_context(at::DeviceType t);

template <at::DeviceType t>
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
