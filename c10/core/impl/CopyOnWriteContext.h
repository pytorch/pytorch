#pragma once

#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/intrusive_ptr.h>

#include <atomic>

namespace c10 {
namespace impl {

// MUST be dynamically allocated; static allocation not permissible
class C10_API CopyOnWriteContext {
  void* data_;
  DeleterFnPtr deleter_;
  mutable std::atomic<size_t> refcount_;
  CopyOnWriteContext(void* data, DeleterFnPtr deleter)
      : data_(data), deleter_(deleter), refcount_(1) {}
  friend C10_API std::unique_ptr<CopyOnWriteContext, DeleterFnPtr> newCopyOnWriteContext(
      void* data,
      DeleterFnPtr deleter);

 public:
  void incref() {
    detail::atomic_refcount_increment(refcount_);
  }
  void decref() {
    size_t r = detail::atomic_refcount_decrement(refcount_);
    if (r == 0) {
      delete this;
    }
  }
  ~CopyOnWriteContext() {
    (*deleter_)(data_);
  }
};

C10_API void deleteCopyOnWriteContext(void* ctx);
C10_API inline std::unique_ptr<CopyOnWriteContext, DeleterFnPtr>
newCopyOnWriteContext(void* data, DeleterFnPtr deleter) {
  return std::unique_ptr<CopyOnWriteContext, DeleterFnPtr>(
      new CopyOnWriteContext(data, deleter), &deleteCopyOnWriteContext);
}

} // namespace impl
} // namespace c10
