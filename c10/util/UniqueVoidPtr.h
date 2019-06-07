#pragma once
#include <memory>

#include <c10/macros/Macros.h>

namespace c10 {

using DeleterFnPtr = void (*)(void*);

namespace detail {

// Does not delete anything
CAFFE2_API void deleteNothing(void*);

// A detail::UniqueVoidPtr is an owning smart pointer like unique_ptr, but
// with three major differences:
//
//    1) It is specialized to void
//
//    2) It is specialized for a function pointer deleter
//       void(void* ctx); i.e., the deleter doesn't take a
//       reference to the data, just to a context pointer
//       (erased as void*).  In fact, internally, this pointer
//       is implemented as having an owning reference to
//       context, and a non-owning reference to data; this is why
//       you release_context(), not release() (the conventional
//       API for release() wouldn't give you enough information
//       to properly dispose of the object later.)
//
//    3) The deleter is guaranteed to be called when the unique
//       pointer is destructed and the context is non-null; this is different
//       from std::unique_ptr where the deleter is not called if the
//       data pointer is null.
//
// Some of the methods have slightly different types than std::unique_ptr
// to reflect this.
//
class UniqueVoidPtr {
 private:
  // Lifetime tied to ctx_
  void* data_;
  std::unique_ptr<void, DeleterFnPtr> ctx_;

 public:
  UniqueVoidPtr() : data_(nullptr), ctx_(nullptr, &deleteNothing) {}
  explicit UniqueVoidPtr(void* data)
      : data_(data), ctx_(nullptr, &deleteNothing) {}
  UniqueVoidPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter)
      : data_(data), ctx_(ctx, ctx_deleter ? ctx_deleter : &deleteNothing) {}
  void* operator->() const {
    return data_;
  }
  void clear() {
    ctx_ = nullptr;
    data_ = nullptr;
  }
  void* get() const {
    return data_;
  }
  void* get_context() const {
    return ctx_.get();
  }
  void* release_context() {
    return ctx_.release();
  }
  std::unique_ptr<void, DeleterFnPtr>&& move_context() {
    return std::move(ctx_);
  }
  C10_NODISCARD bool compare_exchange_deleter(DeleterFnPtr expected_deleter, DeleterFnPtr new_deleter) {
    if (get_deleter() != expected_deleter) return false;
    ctx_ = std::unique_ptr<void, DeleterFnPtr>(ctx_.release(), new_deleter);
    return true;
  }

  template <typename T>
  T* cast_context(DeleterFnPtr expected_deleter) const {
    if (get_deleter() != expected_deleter)
      return nullptr;
    return static_cast<T*>(get_context());
  }
  operator bool() const {
    return data_ || ctx_;
  }
  DeleterFnPtr get_deleter() const {
    return ctx_.get_deleter();
  }
};

// Note [How UniqueVoidPtr is implemented]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// UniqueVoidPtr solves a common problem for allocators of tensor data, which
// is that the data pointer (e.g., float*) which you are interested in, is not
// the same as the context pointer (e.g., DLManagedTensor) which you need
// to actually deallocate the data.  Under a conventional deleter design, you
// have to store extra context in the deleter itself so that you can actually
// delete the right thing.  Implementing this with standard C++ is somewhat
// error-prone: if you use a std::unique_ptr to manage tensors, the deleter will
// not be called if the data pointer is nullptr, which can cause a leak if the
// context pointer is non-null (and the deleter is responsible for freeing both
// the data pointer and the context pointer).
//
// So, in our reimplementation of unique_ptr, which just store the context
// directly in the unique pointer, and attach the deleter to the context
// pointer itself.  In simple cases, the context pointer is just the pointer
// itself.

inline bool operator==(const UniqueVoidPtr& sp, std::nullptr_t) noexcept {
  return !sp;
}
inline bool operator==(std::nullptr_t, const UniqueVoidPtr& sp) noexcept {
  return !sp;
}
inline bool operator!=(const UniqueVoidPtr& sp, std::nullptr_t) noexcept {
  return sp;
}
inline bool operator!=(std::nullptr_t, const UniqueVoidPtr& sp) noexcept {
  return sp;
}

} // namespace detail
} // namespace c10
