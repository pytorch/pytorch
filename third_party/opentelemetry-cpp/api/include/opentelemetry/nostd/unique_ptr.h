// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(OPENTELEMETRY_STL_VERSION)
#  if OPENTELEMETRY_STL_VERSION >= 2011
#    include "opentelemetry/std/unique_ptr.h"
#    define OPENTELEMETRY_HAVE_STD_UNIQUE_PTR
#  endif
#endif

#if !defined(OPENTELEMETRY_HAVE_STD_UNIQUE_PTR)
#  include <cstddef>
#  include <memory>
#  include <type_traits>
#  include <utility>

#  include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
namespace detail
{
template <class T>
struct unique_ptr_element_type
{
  using type = T;
};

template <class T>
struct unique_ptr_element_type<T[]>
{
  using type = T;
};
}  // namespace detail

/**
 * Provide a simplified port of std::unique_ptr that has ABI stability.
 *
 * Note: This implementation doesn't allow for a custom deleter.
 */
template <class T>
class unique_ptr
{
public:
  using element_type = typename detail::unique_ptr_element_type<T>::type;
  using pointer      = element_type *;

  unique_ptr() noexcept : ptr_{nullptr} {}

  unique_ptr(std::nullptr_t) noexcept : ptr_{nullptr} {}

  explicit unique_ptr(pointer ptr) noexcept : ptr_{ptr} {}

  unique_ptr(unique_ptr &&other) noexcept : ptr_{other.release()} {}

  template <class U,
            typename std::enable_if<std::is_convertible<U *, pointer>::value>::type * = nullptr>
  unique_ptr(unique_ptr<U> &&other) noexcept : ptr_{other.release()}
  {}

  template <class U,
            typename std::enable_if<std::is_convertible<U *, pointer>::value>::type * = nullptr>
  unique_ptr(std::unique_ptr<U> &&other) noexcept : ptr_{other.release()}
  {}

  ~unique_ptr() { reset(); }

  unique_ptr &operator=(unique_ptr &&other) noexcept
  {
    reset(other.release());
    return *this;
  }

  unique_ptr &operator=(std::nullptr_t) noexcept
  {
    reset();
    return *this;
  }

  template <class U,
            typename std::enable_if<std::is_convertible<U *, pointer>::value>::type * = nullptr>
  unique_ptr &operator=(unique_ptr<U> &&other) noexcept
  {
    reset(other.release());
    return *this;
  }

  template <class U,
            typename std::enable_if<std::is_convertible<U *, pointer>::value>::type * = nullptr>
  unique_ptr &operator=(std::unique_ptr<U> &&other) noexcept
  {
    reset(other.release());
    return *this;
  }

  operator std::unique_ptr<T>() &&noexcept { return std::unique_ptr<T>{release()}; }

  operator bool() const noexcept { return ptr_ != nullptr; }

  element_type &operator*() const noexcept { return *ptr_; }

  pointer operator->() const noexcept { return get(); }

  pointer get() const noexcept { return ptr_; }

  void reset(pointer ptr = nullptr) noexcept
  {
    if (ptr_ != nullptr)
    {
      this->delete_ptr();
    }
    ptr_ = ptr;
  }

  pointer release() noexcept
  {
    auto result = ptr_;
    ptr_        = nullptr;
    return result;
  }

  void swap(unique_ptr &other) noexcept { std::swap(ptr_, other.ptr_); }

private:
  pointer ptr_;

  void delete_ptr() noexcept
  {
    if (std::is_array<T>::value)
    {
      delete[] ptr_;
    }
    else
    {
      delete ptr_;
    }
  }
};

template <class T1, class T2>
bool operator==(const unique_ptr<T1> &lhs, const unique_ptr<T2> &rhs) noexcept
{
  return lhs.get() == rhs.get();
}

template <class T>
bool operator==(const unique_ptr<T> &lhs, std::nullptr_t) noexcept
{
  return lhs.get() == nullptr;
}

template <class T>
bool operator==(std::nullptr_t, const unique_ptr<T> &rhs) noexcept
{
  return nullptr == rhs.get();
}

template <class T1, class T2>
bool operator!=(const unique_ptr<T1> &lhs, const unique_ptr<T2> &rhs) noexcept
{
  return lhs.get() != rhs.get();
}

template <class T>
bool operator!=(const unique_ptr<T> &lhs, std::nullptr_t) noexcept
{
  return lhs.get() != nullptr;
}

template <class T>
bool operator!=(std::nullptr_t, const unique_ptr<T> &rhs) noexcept
{
  return nullptr != rhs.get();
}
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
#endif /* OPENTELEMETRY_HAVE_STD_UNIQUE_PTR */
