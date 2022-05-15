#pragma once

#include <memory>
#include <type_traits>

#include <c10/util/Exception.h>
#include <c10/util/MaybeOwned.h>

namespace c10 {

// Compatibility wrapper around a raw pointer so that existing code
// written to deal with a shared_ptr can keep working.
template <typename T>
class SingletonTypePtr {
 public:
  /* implicit */ SingletonTypePtr(T* p) : repr_(p) {}

  // We need this to satisfy Pybind11, but it shouldn't be hit.
  explicit SingletonTypePtr(std::shared_ptr<T>) { TORCH_CHECK(false); }

  using element_type = typename std::shared_ptr<T>::element_type;

  template <typename U = T, std::enable_if_t<!std::is_same<std::remove_const_t<U>, void>::value, bool> = true>
  T& operator*() const {
    return *repr_;
  }

  T* get() const {
    return repr_;
  }

  T* operator->() const {
    return repr_;
  }

  operator bool() const {
    return repr_ != nullptr;
  }

 private:
  T* repr_;
};

template <typename T, typename U>
bool operator==(SingletonTypePtr<T> lhs, SingletonTypePtr<U> rhs) {
  return (void*)lhs.get() == (void*)rhs.get();
}

template <typename T, typename U>
bool operator!=(SingletonTypePtr<T> lhs, SingletonTypePtr<U> rhs) {
  return !(lhs == rhs);
}

} // namespace c10
