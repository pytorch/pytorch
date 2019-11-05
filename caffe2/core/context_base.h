#pragma once

#include <array>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <unordered_map>

#include <c10/macros/Macros.h>
#include <c10/core/Allocator.h>
#include <c10/util/typeid.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>
#include <c10/core/CopyBytes.h>

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {
class Event;

} // namespace caffe2
namespace at {

class BaseContext;

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

  virtual Device device() const = 0;

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
using at::CreateContext;
} // namespace caffe2
