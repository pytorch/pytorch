// Copyright (C) 2011 - 2012 Andrzej Krzemienski.
//
// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// The idea and interface is based on Boost.Optional library
// authored by Fernando Luis Cacciola Carballal
//
// From https://github.com/akrzemi1/Optional
//
// C10
// - Move file to `c10` namespace.
// - Remove macro use in line 478 because the nvcc device compiler cannot handle
// it.
// - Revise constructor logic so that it is 1) consistent with c++ 17 standard
// documented here in (8):
// https://en.cppreference.com/w/cpp/utility/optional/optional, and 2) able to
// support initialization of optionals from convertible type U.
// - Remove the constructors for `optional(const T&)` and `optional(T&&)`, as
// they can be handled by the template<U=T> case with the default template
// argument.
// - Move `constexpr struct in_place_t {} in_place{}` to `c10/util/in_place.h`
// so that it can also be used in `c10/util/variant.h`.
// - Remove special cases for pre-c++14 compilers to make code simpler.

#ifndef C10_UTIL_OPTIONAL_H_
#define C10_UTIL_OPTIONAL_H_

#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/in_place.h>

#include <cassert>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <c10/util/C++17.h>
#include <c10/util/Metaprogramming.h>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 4624) // destructor was implicitly defined as deleted
#endif

#define TR2_OPTIONAL_REQUIRES(...) \
  typename std::enable_if<__VA_ARGS__::value, bool>::type = false

namespace c10 {

// 20.5.4, optional for object types
template <class T>
class optional;

// 20.5.5, optional for lvalue reference types
template <class T>
class optional<T&>;

// workaround: std utility functions aren't constexpr yet
template <class T>
inline constexpr T&& constexpr_forward(
    typename std::remove_reference<T>::type& t) noexcept {
  return static_cast<T&&>(t);
}

template <class T>
inline constexpr T&& constexpr_forward(
    typename std::remove_reference<T>::type&& t) noexcept {
  static_assert(!std::is_lvalue_reference<T>::value, "!!");
  return static_cast<T&&>(t);
}

template <class T>
inline constexpr typename std::remove_reference<T>::type&& constexpr_move(
    T&& t) noexcept {
  return static_cast<typename std::remove_reference<T>::type&&>(t);
}

#if defined NDEBUG
#define TR2_OPTIONAL_ASSERTED_EXPRESSION(CHECK, EXPR) (EXPR)
#else
#define TR2_OPTIONAL_ASSERTED_EXPRESSION(CHECK, EXPR) \
  ((CHECK) ? (EXPR) : ([] { assert(false); }(), (EXPR)))
#endif

#if defined(__CUDA_ARCH__)
#define TR2_OPTIONAL_HOST_CONSTEXPR
#else
#define TR2_OPTIONAL_HOST_CONSTEXPR constexpr
#endif

// Sphinx chokes on static_addressof, so exclude it from Doxygen
// generation.  See https://github.com/sphinx-doc/sphinx/issues/7944
// \cond

namespace detail_ {

// VS doesn't handle constexpr well, so we need to skip these stuff.
#if (defined _MSC_VER)
template <typename T>
T* static_addressof(T& ref) {
  return std::addressof(ref);
}
#else
// static_addressof: a constexpr version of addressof
template <typename T>
struct has_overloaded_addressof {
  template <class X>
  constexpr static bool has_overload(...) {
    return false;
  }

  template <class X, size_t S = sizeof(std::declval<X&>().operator&())>
  constexpr static bool has_overload(bool) {
    return true;
  }

  constexpr static bool value = has_overload<T>(true);
};

template <typename T, TR2_OPTIONAL_REQUIRES(!has_overloaded_addressof<T>)>
constexpr T* static_addressof(T& ref) {
  return &ref;
}

template <typename T, TR2_OPTIONAL_REQUIRES(has_overloaded_addressof<T>)>
T* static_addressof(T& ref) {
  return std::addressof(ref);
}
#endif

// the call to convert<A>(b) has return type A and converts b to type A iff b
// decltype(b) is implicitly convertible to A
template <class U>
constexpr U convert(U v) {
  return v;
}

} // namespace detail_

// \endcond

constexpr struct trivial_init_t {
} trivial_init{};

// 20.5.7, Disengaged state indicator
struct nullopt_t {
  constexpr explicit nullopt_t(int) {}
};
constexpr nullopt_t nullopt{0};

// 20.5.8, class bad_optional_access
class bad_optional_access : public std::logic_error {
 public:
  explicit bad_optional_access(const std::string& what_arg)
      : logic_error{what_arg} {}
  explicit bad_optional_access(const char* what_arg) : logic_error{what_arg} {}
};

template <class T>
union storage_t {
  unsigned char dummy_{};
  T value_;

#if __cplusplus >= 202002L
  constexpr
#endif
      storage_t(trivial_init_t) noexcept {
    new (&dummy_) unsigned char;
  }

  template <class... Args>
  constexpr storage_t(Args&&... args)
      : value_(constexpr_forward<Args>(args)...) {}

  ~storage_t() {}
};

template <class T>
union constexpr_storage_t {
  unsigned char dummy_;
  T value_;

#if __cplusplus >= 202002L
  // C++20 lifted the requirement to initialize a union member in order to be
  // constexpr.
  constexpr constexpr_storage_t(trivial_init_t) noexcept {
    new (&dummy_) unsigned char;
  }
#else
  constexpr constexpr_storage_t(trivial_init_t) noexcept : dummy_() {}
#endif

  template <class... Args>
  constexpr constexpr_storage_t(Args&&... args)
      : value_(constexpr_forward<Args>(args)...) {}

  constexpr constexpr_storage_t(const constexpr_storage_t&) = default;
  constexpr constexpr_storage_t& operator=(const constexpr_storage_t&) =
      default;
  ~constexpr_storage_t() = default;
};

template <class T>
struct optional_base {
  bool init_;
  storage_t<T> storage_;

  constexpr optional_base() noexcept : init_(false), storage_(trivial_init){};

  explicit constexpr optional_base(const optional_base<T>& v)
      : init_(v.init_), storage_(trivial_init) {
    if (init_) {
      ::new (dataptr()) T(v.storage_.value_);
    }
  }

  explicit constexpr optional_base(const T& v) : init_(true), storage_(v) {}

  explicit constexpr optional_base(optional_base<T>&& v) noexcept(
      std::is_nothrow_move_constructible<T>::value)
      : init_(v.init_), storage_(trivial_init) {
    if (init_) {
      ::new (dataptr()) T(std::move(v.storage_.value_));
    }
  }

  explicit constexpr optional_base(T&& v)
      : init_(true), storage_(constexpr_move(v)) {}

  template <class... Args>
  explicit optional_base(in_place_t, Args&&... args)
      : init_(true), storage_(constexpr_forward<Args>(args)...) {}

  template <
      class U,
      class... Args,
      TR2_OPTIONAL_REQUIRES(std::is_constructible<T, std::initializer_list<U>>)>
  explicit optional_base(
      in_place_t,
      std::initializer_list<U> il,
      Args&&... args)
      : init_(true), storage_(il, std::forward<Args>(args)...) {}

  optional_base& operator=(const optional_base& rhs) {
    if (init_ && !rhs.init_) {
      clear();
    } else if (!init_ && rhs.init_) {
      init_ = true;
      ::new (dataptr()) T(rhs.storage_.value_);
    } else if (init_ && rhs.init_) {
      storage_.value_ = rhs.storage_.value_;
    }
    return *this;
  }

  optional_base& operator=(optional_base&& rhs) noexcept(
      std::is_nothrow_move_assignable<T>::value&&
          std::is_nothrow_move_constructible<T>::value) {
    if (init_ && !rhs.init_) {
      clear();
    } else if (!init_ && rhs.init_) {
      init_ = true;
      ::new (dataptr()) T(std::move(rhs.storage_.value_));
    } else if (init_ && rhs.init_) {
      storage_.value_ = std::move(rhs.storage_.value_);
    }
    return *this;
  }

  ~optional_base() {
    if (init_)
      storage_.value_.T::~T();
  }

  constexpr bool initialized() const noexcept {
    return init_;
  }

  void setInitialized(bool init) noexcept {
    init_ = init;
  }

 private:
  typename std::remove_const<T>::type* dataptr() {
    return std::addressof(storage_.value_);
  }

  constexpr const T* dataptr() const {
    return detail_::static_addressof(storage_.value_);
  }

  void clear() noexcept {
    if (init_) {
      dataptr()->~T();
    }
    init_ = false;
  }
};

template <class T>
struct constexpr_optional_base {
  bool init_;
  constexpr_storage_t<T> storage_;

  constexpr constexpr_optional_base() noexcept
      : init_(false), storage_(trivial_init){};

  explicit constexpr constexpr_optional_base(
      const constexpr_optional_base<T>& v)
      : init_(v.init_), storage_(trivial_init) {
    if (init_) {
      ::new (dataptr()) T(v.storage_.value_);
    }
  }

  explicit constexpr constexpr_optional_base(
      constexpr_optional_base<T>&&
          v) noexcept(std::is_nothrow_move_constructible<T>::value)
      : init_(v.init_), storage_(trivial_init) {
    if (init_) {
      ::new (dataptr()) T(std::move(v.storage_.value_));
    }
  }

  explicit constexpr constexpr_optional_base(const T& v)
      : init_(true), storage_(v) {}

  explicit constexpr constexpr_optional_base(T&& v)
      : init_(true), storage_(constexpr_move(v)) {}

  template <class... Args>
  explicit constexpr constexpr_optional_base(in_place_t, Args&&... args)
      : init_(true), storage_(constexpr_forward<Args>(args)...) {}

  template <
      class U,
      class... Args,
      TR2_OPTIONAL_REQUIRES(std::is_constructible<T, std::initializer_list<U>>)>
  constexpr explicit constexpr_optional_base(
      in_place_t,
      std::initializer_list<U> il,
      Args&&... args)
      : init_(true), storage_(il, std::forward<Args>(args)...) {}

  ~constexpr_optional_base() = default;

  constexpr_optional_base& operator=(const constexpr_optional_base& rhs) {
    if (init_ && !rhs.init_) {
      clear();
    } else if (!init_ && rhs.init_) {
      init_ = true;
      ::new (dataptr()) T(rhs.storage_.value_);
    } else if (init_ && rhs.init_) {
      storage_.value_ = rhs.storage_.value_;
    }
    return *this;
  }

  constexpr_optional_base& operator=(constexpr_optional_base&& rhs) noexcept(
      std::is_nothrow_move_assignable<T>::value&&
          std::is_nothrow_move_constructible<T>::value) {
    if (init_ && !rhs.init_) {
      clear();
    } else if (!init_ && rhs.init_) {
      init_ = true;
      ::new (dataptr()) T(std::move(rhs.storage_.value_));
    } else if (init_ && rhs.init_) {
      storage_.value_ = std::move(rhs.storage_.value_);
    }
    return *this;
  }

  constexpr bool initialized() const noexcept {
    return init_;
  }
  void setInitialized(bool init) noexcept {
    init_ = init;
  }

 private:
  typename std::remove_const<T>::type* dataptr() {
    return std::addressof(storage_.value_);
  }

  constexpr const T* dataptr() const {
    return detail_::static_addressof(storage_.value_);
  }

  void clear() noexcept {
    init_ = false;
  }
};

// HACK: Optimization for trivially copyable types. The mainline
// implementation fails to have trivial copy/move operations in these
// cases, and we care about them, so just implement that directly.
template <class T>
struct trivially_copyable_optimization_optional_base {
  bool init_;
  constexpr_storage_t<T> storage_;

  constexpr trivially_copyable_optimization_optional_base() noexcept
      : init_(false), storage_(trivial_init) {}

  explicit constexpr trivially_copyable_optimization_optional_base(const T& v)
      : init_(true), storage_(v) {}

  explicit constexpr trivially_copyable_optimization_optional_base(T&& v)
      : init_(true), storage_(constexpr_move(v)) {}

  template <class... Args>
  explicit constexpr trivially_copyable_optimization_optional_base(
      in_place_t,
      Args&&... args)
      : init_(true), storage_(constexpr_forward<Args>(args)...) {}

  template <
      class U,
      class... Args,
      TR2_OPTIONAL_REQUIRES(std::is_constructible<T, std::initializer_list<U>>)>
  constexpr explicit trivially_copyable_optimization_optional_base(
      in_place_t,
      std::initializer_list<U> il,
      Args&&... args)
      : init_(true), storage_(il, std::forward<Args>(args)...) {}

  constexpr trivially_copyable_optimization_optional_base(
      const trivially_copyable_optimization_optional_base&) = default;

  constexpr trivially_copyable_optimization_optional_base& operator=(
      const trivially_copyable_optimization_optional_base&) = default;
  ~trivially_copyable_optimization_optional_base() = default;

  constexpr bool initialized() const noexcept {
    return init_;
  }
  void setInitialized(bool init) noexcept {
    init_ = init;
  }
};

// HACK: Optimization for ArrayRef<T>. We take advantage of an unused
// bit pattern in ArrayRef (inspired by Arthur O'Dwyer's
// tombstone_traits -- see https://youtu.be/MWBfmmg8-Yo?t=2466) to
// keep the size of c10::optional::ArrayRef<T> down to 16 bytes, which
// allows it to be passed to functions in registers instead of getting
// passed in memory per item 5c of the classification algorithm in
// section 3.2.3 of the System V ABI document
// (https://www.uclibc.org/docs/psABI-x86_64.pdf).
template <class ArrayRefT>
class arrayref_optional_base {
 public:
  union storage {
    struct raw {
      // ArrayRef has the invariant that if Data is nullptr then
      // Length must be zero, so this is an unused bit pattern.
      const void* p = nullptr;
      size_t sz = 1;
    } uninitialized_{};
    ArrayRefT value_;

    constexpr storage() noexcept : uninitialized_() {
      setUninitialized();
    }

    constexpr void setUninitialized() noexcept {
      uninitialized_.p = nullptr;
      uninitialized_.sz = 1;
    }

    explicit constexpr storage(ArrayRefT& v) : value_(v) {}

    template <typename T>
    explicit constexpr storage(const std::initializer_list<T>& v) : value_(v) {}

    template <class... Args>
    explicit constexpr storage(Args&&... args)
        : value_(constexpr_forward<Args>(args)...) {}
  };

  storage storage_;

  constexpr arrayref_optional_base() noexcept = default;

  explicit constexpr arrayref_optional_base(const ArrayRefT& v) : storage_(v) {}

  template <class... Args>
  explicit constexpr arrayref_optional_base(in_place_t, Args&&... args)
      : storage_(constexpr_forward<Args>(args)...) {}

  template <typename T>
  explicit constexpr arrayref_optional_base(
      in_place_t,
      const std::initializer_list<T>& v)
      : storage_(v) {}

  constexpr bool initialized() const noexcept {
    return storage_.uninitialized_.p != nullptr ||
        storage_.uninitialized_.sz == 0;
  }

  void setInitialized(bool init) noexcept {
    if (!init) {
      storage_.setUninitialized();
    } else {
      assert(initialized());
    }
  }
};

namespace detail_ {
template <typename T>
struct is_arrayref : std::false_type {};

template <typename T>
struct is_arrayref<c10::ArrayRef<T>> : std::true_type {};
} // namespace detail_

template <class T>
using OptionalBase = std::conditional_t<
    detail_::is_arrayref<T>::value,
    arrayref_optional_base<T>,
    std::conditional_t<
        std::is_trivially_destructible<T>::value &&
            C10_IS_TRIVIALLY_COPYABLE(T) &&
            // Avoid using is_trivially_copy_{constructible,assignable}
            // because old GCC versions don't support them. Also,
            // is_trivially_copyable seems not to do what I expect, so check
            // trivially_copyable_optimization_optional_base directly.
            std::is_copy_constructible<
                trivially_copyable_optimization_optional_base<T>>::value &&
            std::is_copy_assignable<
                trivially_copyable_optimization_optional_base<T>>::value,
        trivially_copyable_optimization_optional_base<T>,
        std::conditional_t<
            std::is_trivially_destructible<T>::value, // if possible
            constexpr_optional_base<std::remove_const_t<T>>, // use base with
                                                             // trivial
                                                             // destructor
            optional_base<std::remove_const_t<T>>>>>;

template <class T>
class optional : private OptionalBase<T> {
  template <class U> // re-declaration for nvcc on Windows.
  using OptionalBase = std::conditional_t<
      detail_::is_arrayref<U>::value,
      arrayref_optional_base<U>,
      std::conditional_t<
          std::is_trivially_destructible<U>::value &&
              C10_IS_TRIVIALLY_COPYABLE(U) &&
              // Avoid using is_trivially_copy_{constructible,assignable}
              // because old GCC versions don't support them. Also,
              // is_trivially_copyable seems not to do what I expect, so
              // check trivially_copyable_optimization_optional_base
              // directly.
              std::is_copy_constructible<
                  trivially_copyable_optimization_optional_base<U>>::value &&
              std::is_copy_assignable<
                  trivially_copyable_optimization_optional_base<U>>::value,
          trivially_copyable_optimization_optional_base<U>,
          std::conditional_t<
              std::is_trivially_destructible<U>::value, // if possible
              constexpr_optional_base<std::remove_const_t<U>>, // use base
                                                               // with
                                                               // trivial
                                                               // destructor
              optional_base<std::remove_const_t<U>>>>>;

  static_assert(
      !std::is_same<typename std::decay<T>::type, nullopt_t>::value,
      "bad T");
  static_assert(
      !std::is_same<typename std::decay<T>::type, in_place_t>::value,
      "bad T");

  constexpr bool initialized() const noexcept {
    return OptionalBase<T>::initialized();
  }
  typename std::remove_const<T>::type* dataptr() {
    return std::addressof(OptionalBase<T>::storage_.value_);
  }
  constexpr const T* dataptr() const {
    return detail_::static_addressof(OptionalBase<T>::storage_.value_);
  }

  constexpr const T& contained_val() const& {
    return OptionalBase<T>::storage_.value_;
  }
  constexpr T&& contained_val() && {
    return std::move(OptionalBase<T>::storage_.value_);
  }
  constexpr T& contained_val() & {
    return OptionalBase<T>::storage_.value_;
  }

  void clear() noexcept {
    if (initialized())
      dataptr()->~T();
    OptionalBase<T>::setInitialized(false);
  }

  template <class... Args>
  void initialize(Args&&... args) noexcept(
      noexcept(T(std::forward<Args>(args)...))) {
    assert(!initialized());
    ::new (static_cast<void*>(dataptr())) T(std::forward<Args>(args)...);
    OptionalBase<T>::setInitialized(true);
  }

  template <class U, class... Args>
  void initialize(std::initializer_list<U> il, Args&&... args) noexcept(
      noexcept(T(il, std::forward<Args>(args)...))) {
    assert(!initialized());
    ::new (static_cast<void*>(dataptr())) T(il, std::forward<Args>(args)...);
    OptionalBase<T>::setInitialized(true);
  }

 public:
  typedef T value_type;

  // 20.5.5.1, constructors
  constexpr optional() noexcept = default;
  constexpr optional(nullopt_t) noexcept : OptionalBase<T>(){};

  optional(const optional& rhs) = default;
  optional(optional&& rhs) = default;

  // see https://github.com/akrzemi1/Optional/issues/16
  // and https://en.cppreference.com/w/cpp/utility/optional/optional,
  // in constructor 8, the std::optional spec can allow initialization
  // of optionals from convertible type U
  //
  // 8 - implicit move construct from value
  template <
      typename U = T,
      TR2_OPTIONAL_REQUIRES(
          std::is_constructible<T, U&&>::value &&
          !std::is_same<typename std::decay<U>::type, in_place_t>::value &&
          !std::is_same<typename std::decay<U>::type, optional<T>>::value &&
          std::is_convertible<U&&, T>)>
  constexpr optional(U&& u) : OptionalBase<T>(std::forward<U>(u)) {}

  // 8 - explicit move construct from value
  template <
      typename U = T,
      TR2_OPTIONAL_REQUIRES(
          std::is_constructible<T, U&&>::value &&
          !std::is_same<typename std::decay<U>::type, in_place_t>::value &&
          !std::is_same<typename std::decay<U>::type, optional<T>>::value &&
          !std::is_convertible<U&&, T>)>
  explicit constexpr optional(U&& u) : OptionalBase<T>(std::forward<U>(u)) {}

  template <class... Args>
  explicit constexpr optional(in_place_t, Args&&... args)
      : OptionalBase<T>(in_place_t{}, constexpr_forward<Args>(args)...) {}

  template <
      class U,
      class... Args,
      TR2_OPTIONAL_REQUIRES(std::is_constructible<T, std::initializer_list<U>>)>
  constexpr explicit optional(
      in_place_t,
      std::initializer_list<U> il,
      Args&&... args)
      : OptionalBase<T>(in_place_t{}, il, constexpr_forward<Args>(args)...) {}

  // 20.5.4.2, Destructor
  ~optional() = default;

  // 20.5.4.3, assignment
  optional& operator=(nullopt_t) noexcept {
    clear();
    return *this;
  }

  optional& operator=(const optional& rhs) = default;

  optional& operator=(optional&& rhs) = default;

  template <
      class U = T,
      typename = std::enable_if_t<
          std::is_constructible<T, U>::value &&
          !std::is_same<typename std::decay<U>::type, optional<T>>::value &&
          (std::is_scalar<T>::value ||
           std::is_same<typename std::decay<U>::type, T>::value) &&
          std::is_assignable<T&, U>::value>>
  optional& operator=(U&& v) {
    if (initialized()) {
      contained_val() = std::forward<U>(v);
    } else {
      initialize(std::forward<U>(v));
    }
    return *this;
  }

  template <class... Args>
  void emplace(Args&&... args) {
    clear();
    initialize(std::forward<Args>(args)...);
  }

  template <class U, class... Args>
  void emplace(std::initializer_list<U> il, Args&&... args) {
    clear();
    initialize<U, Args...>(il, std::forward<Args>(args)...);
  }

  // 20.5.4.4, Swap
  void swap(optional<T>& rhs) noexcept(
      std::is_nothrow_move_constructible<T>::value&& noexcept(
          std::swap(std::declval<T&>(), std::declval<T&>()))) {
    if (initialized() == true && rhs.initialized() == false) {
      rhs.initialize(std::move(**this));
      clear();
    } else if (initialized() == false && rhs.initialized() == true) {
      initialize(std::move(*rhs));
      rhs.clear();
    } else if (initialized() == true && rhs.initialized() == true) {
      using std::swap;
      swap(**this, *rhs);
    }
  }

  // 20.5.4.5, Observers

  explicit constexpr operator bool() const noexcept {
    return initialized();
  }
  constexpr bool has_value() const noexcept {
    return initialized();
  }

  TR2_OPTIONAL_HOST_CONSTEXPR T const* operator->() const {
    return TR2_OPTIONAL_ASSERTED_EXPRESSION(initialized(), dataptr());
  }

  TR2_OPTIONAL_HOST_CONSTEXPR T* operator->() {
    assert(initialized());
    return dataptr();
  }

  TR2_OPTIONAL_HOST_CONSTEXPR T const& operator*() const& {
    return TR2_OPTIONAL_ASSERTED_EXPRESSION(initialized(), contained_val());
  }

  TR2_OPTIONAL_HOST_CONSTEXPR T& operator*() & {
    assert(initialized());
    return contained_val();
  }

  TR2_OPTIONAL_HOST_CONSTEXPR T&& operator*() && {
    assert(initialized());
    return constexpr_move(contained_val());
  }

  TR2_OPTIONAL_HOST_CONSTEXPR T const& value() const& {
    return initialized()
        ? contained_val()
        : (throw bad_optional_access("bad optional access"), contained_val());
  }

  TR2_OPTIONAL_HOST_CONSTEXPR T& value() & {
    return initialized()
        ? contained_val()
        : (throw bad_optional_access("bad optional access"), contained_val());
  }

  TR2_OPTIONAL_HOST_CONSTEXPR T&& value() && {
    if (!initialized())
      throw bad_optional_access("bad optional access");
    return std::move(contained_val());
  }

  template <class V>
  constexpr T value_or(V&& v) const& {
    return *this ? **this : detail_::convert<T>(constexpr_forward<V>(v));
  }

  template <class V>
  constexpr T value_or(V&& v) && {
    return *this ? constexpr_move(*this).contained_val()
                 : detail_::convert<T>(constexpr_forward<V>(v));
  }

  // 20.6.3.6, modifiers
  void reset() noexcept {
    clear();
  }
};

template <class T, class F>
constexpr T value_or_else(const optional<T>& v, F&& func) {
  static_assert(
      std::is_convertible<typename std::invoke_result_t<F>, T>::value,
      "func parameters must be a callable that returns a type convertible to the value stored in the optional");
  return v.has_value() ? *v : detail_::convert<T>(std::forward<F>(func)());
}

template <class T, class F>
constexpr T value_or_else(optional<T>&& v, F&& func) {
  static_assert(
      std::is_convertible<typename std::invoke_result_t<F>, T>::value,
      "func parameters must be a callable that returns a type convertible to the value stored in the optional");
  return v.has_value() ? constexpr_move(std::move(v).contained_val())
                       : detail_::convert<T>(std::forward<F>(func)());
}

// XXX: please refrain from using optional<T&>, since it is being against with
// the optional standard in c++ 17, see the debate and the details here:
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3406#rationale.refs
// if you need it, consider using optional<std::reference_wrapper<T>> or *
// pointer
//
// we leave the implementation here in case we want to reconsider using it in
// the future if it becomes a definitely necessary case.
template <class T>
class optional<T&> {
  // add this assert to prevent user from using optional reference as indicated
  // above
  static_assert(
      sizeof(T) == 0,
      "optional references is ill-formed, \
    consider use optional of a std::reference_wrapper of type T to \
    hold a reference if you really need to");

  static_assert(!std::is_same<T, nullopt_t>::value, "bad T");
  static_assert(!std::is_same<T, in_place_t>::value, "bad T");
  T* ref;

 public:
  // 20.5.5.1, construction/destruction
  constexpr optional() noexcept : ref(nullptr) {}

  constexpr optional(nullopt_t) noexcept : ref(nullptr) {}

  template <typename U = T>
  constexpr optional(U& u) noexcept : ref(detail_::static_addressof(u)) {}

  template <typename U = T>
  optional(U&&) = delete;

  constexpr optional(const optional& rhs) noexcept : ref(rhs.ref) {}

  explicit constexpr optional(in_place_t, T& v) noexcept
      : ref(detail_::static_addressof(v)) {}

  explicit optional(in_place_t, T&&) = delete;

  ~optional() = default;

  // 20.5.5.2, mutation
  optional& operator=(nullopt_t) noexcept {
    ref = nullptr;
    return *this;
  }

  // optional& operator=(const optional& rhs) noexcept {
  // ref = rhs.ref;
  // return *this;
  // }

  // optional& operator=(optional&& rhs) noexcept {
  // ref = rhs.ref;
  // return *this;
  // }

  template <
      typename U,
      typename = std::enable_if_t<
          std::is_same_v<typename std::decay<U>::type, optional<T&>>>>
  optional& operator=(U&& rhs) noexcept {
    ref = rhs.ref;
    return *this;
  }

  template <typename U>
  auto operator=(U&& rhs) noexcept -> typename std::enable_if<
      !std::is_same<typename std::decay<U>::type, optional<T&>>::value,
      optional&>::type = delete;

  void emplace(T& v) noexcept {
    ref = detail_::static_addressof(v);
  }

  void emplace(T&&) = delete;

  void swap(optional<T&>& rhs) noexcept {
    std::swap(ref, rhs.ref);
  }

  // 20.5.5.3, observers
  TR2_OPTIONAL_HOST_CONSTEXPR T* operator->() const {
    return TR2_OPTIONAL_ASSERTED_EXPRESSION(ref, ref);
  }

  TR2_OPTIONAL_HOST_CONSTEXPR T& operator*() const {
    return TR2_OPTIONAL_ASSERTED_EXPRESSION(ref, *ref);
  }

  constexpr T& value() const {
    return ref ? *ref
               : (throw bad_optional_access("bad optional access"), *ref);
  }

  explicit constexpr operator bool() const noexcept {
    return ref != nullptr;
  }

  constexpr bool has_value() const noexcept {
    return ref != nullptr;
  }

  template <class V>
  constexpr typename std::decay<T>::type value_or(V&& v) const {
    return *this ? **this
                 : detail_::convert<typename std::decay<T>::type>(
                       constexpr_forward<V>(v));
  }

  // x.x.x.x, modifiers
  void reset() noexcept {
    ref = nullptr;
  }
};

template <class T>
class optional<T&&> {
  static_assert(sizeof(T) == 0, "optional rvalue references disallowed");
};

// 20.5.8, Relational operators
template <class T>
constexpr bool operator==(const optional<T>& x, const optional<T>& y) {
  return bool(x) != bool(y) ? false : bool(x) == false ? true : *x == *y;
}

template <class T>
constexpr bool operator!=(const optional<T>& x, const optional<T>& y) {
  return !(x == y);
}

template <class T>
constexpr bool operator<(const optional<T>& x, const optional<T>& y) {
  return (!y) ? false : (!x) ? true : *x < *y;
}

template <class T>
constexpr bool operator>(const optional<T>& x, const optional<T>& y) {
  return (y < x);
}

template <class T>
constexpr bool operator<=(const optional<T>& x, const optional<T>& y) {
  return !(y < x);
}

template <class T>
constexpr bool operator>=(const optional<T>& x, const optional<T>& y) {
  return !(x < y);
}

// 20.5.9, Comparison with nullopt
template <class T>
constexpr bool operator==(const optional<T>& x, nullopt_t) noexcept {
  return (!x);
}

template <class T>
constexpr bool operator==(nullopt_t, const optional<T>& x) noexcept {
  return (!x);
}

template <class T>
constexpr bool operator!=(const optional<T>& x, nullopt_t) noexcept {
  return bool(x);
}

template <class T>
constexpr bool operator!=(nullopt_t, const optional<T>& x) noexcept {
  return bool(x);
}

template <class T>
constexpr bool operator<(const optional<T>&, nullopt_t) noexcept {
  return false;
}

template <class T>
constexpr bool operator<(nullopt_t, const optional<T>& x) noexcept {
  return bool(x);
}

template <class T>
constexpr bool operator<=(const optional<T>& x, nullopt_t) noexcept {
  return (!x);
}

template <class T>
constexpr bool operator<=(nullopt_t, const optional<T>&) noexcept {
  return true;
}

template <class T>
constexpr bool operator>(const optional<T>& x, nullopt_t) noexcept {
  return bool(x);
}

template <class T>
constexpr bool operator>(nullopt_t, const optional<T>&) noexcept {
  return false;
}

template <class T>
constexpr bool operator>=(const optional<T>&, nullopt_t) noexcept {
  return true;
}

template <class T>
constexpr bool operator>=(nullopt_t, const optional<T>& x) noexcept {
  return (!x);
}

// 20.5.10, Comparison with T
template <class T, class U>
constexpr bool operator==(const optional<T>& x, const U& v) {
  return bool(x) ? *x == v : false;
}

template <class T, class U>
constexpr bool operator==(const U& v, const optional<T>& x) {
  return bool(x) ? v == *x : false;
}

template <class T, class U>
constexpr bool operator!=(const optional<T>& x, const U& v) {
  return bool(x) ? *x != v : true;
}

template <class T, class U>
constexpr bool operator!=(const U& v, const optional<T>& x) {
  return bool(x) ? v != *x : true;
}

template <class T, class U>
constexpr bool operator<(const optional<T>& x, const U& v) {
  return bool(x) ? *x < v : true;
}

template <class T, class U>
constexpr bool operator>(const U& v, const optional<T>& x) {
  return bool(x) ? v > *x : true;
}

template <class T, class U>
constexpr bool operator>(const optional<T>& x, const U& v) {
  return bool(x) ? *x > v : false;
}

template <class T, class U>
constexpr bool operator<(const U& v, const optional<T>& x) {
  return bool(x) ? v < *x : false;
}

template <class T, class U>
constexpr bool operator>=(const optional<T>& x, const U& v) {
  return bool(x) ? *x >= v : false;
}

template <class T, class U>
constexpr bool operator<=(const U& v, const optional<T>& x) {
  return bool(x) ? v <= *x : false;
}

template <class T, class U>
constexpr bool operator<=(const optional<T>& x, const U& v) {
  return bool(x) ? *x <= v : true;
}

template <class T, class U>
constexpr bool operator>=(const U& v, const optional<T>& x) {
  return bool(x) ? v >= *x : true;
}

// Comparison of optional<T&> with T
template <class T>
constexpr bool operator==(const optional<T&>& x, const T& v) {
  return bool(x) ? *x == v : false;
}

template <class T>
constexpr bool operator==(const T& v, const optional<T&>& x) {
  return bool(x) ? v == *x : false;
}

template <class T>
constexpr bool operator!=(const optional<T&>& x, const T& v) {
  return bool(x) ? *x != v : true;
}

template <class T>
constexpr bool operator!=(const T& v, const optional<T&>& x) {
  return bool(x) ? v != *x : true;
}

template <class T>
constexpr bool operator<(const optional<T&>& x, const T& v) {
  return bool(x) ? *x < v : true;
}

template <class T>
constexpr bool operator>(const T& v, const optional<T&>& x) {
  return bool(x) ? v > *x : true;
}

template <class T>
constexpr bool operator>(const optional<T&>& x, const T& v) {
  return bool(x) ? *x > v : false;
}

template <class T>
constexpr bool operator<(const T& v, const optional<T&>& x) {
  return bool(x) ? v < *x : false;
}

template <class T>
constexpr bool operator>=(const optional<T&>& x, const T& v) {
  return bool(x) ? *x >= v : false;
}

template <class T>
constexpr bool operator<=(const T& v, const optional<T&>& x) {
  return bool(x) ? v <= *x : false;
}

template <class T>
constexpr bool operator<=(const optional<T&>& x, const T& v) {
  return bool(x) ? *x <= v : true;
}

template <class T>
constexpr bool operator>=(const T& v, const optional<T&>& x) {
  return bool(x) ? v >= *x : true;
}

// Comparison of optional<T const&> with T
template <class T>
constexpr bool operator==(const optional<const T&>& x, const T& v) {
  return bool(x) ? *x == v : false;
}

template <class T>
constexpr bool operator==(const T& v, const optional<const T&>& x) {
  return bool(x) ? v == *x : false;
}

template <class T>
constexpr bool operator!=(const optional<const T&>& x, const T& v) {
  return bool(x) ? *x != v : true;
}

template <class T>
constexpr bool operator!=(const T& v, const optional<const T&>& x) {
  return bool(x) ? v != *x : true;
}

template <class T>
constexpr bool operator<(const optional<const T&>& x, const T& v) {
  return bool(x) ? *x < v : true;
}

template <class T>
constexpr bool operator>(const T& v, const optional<const T&>& x) {
  return bool(x) ? v > *x : true;
}

template <class T>
constexpr bool operator>(const optional<const T&>& x, const T& v) {
  return bool(x) ? *x > v : false;
}

template <class T>
constexpr bool operator<(const T& v, const optional<const T&>& x) {
  return bool(x) ? v < *x : false;
}

template <class T>
constexpr bool operator>=(const optional<const T&>& x, const T& v) {
  return bool(x) ? *x >= v : false;
}

template <class T>
constexpr bool operator<=(const T& v, const optional<const T&>& x) {
  return bool(x) ? v <= *x : false;
}

template <class T>
constexpr bool operator<=(const optional<const T&>& x, const T& v) {
  return bool(x) ? *x <= v : true;
}

template <class T>
constexpr bool operator>=(const T& v, const optional<const T&>& x) {
  return bool(x) ? v >= *x : true;
}

// 20.5.12, Specialized algorithms
template <class T>
void swap(optional<T>& x, optional<T>& y) noexcept(noexcept(x.swap(y))) {
  x.swap(y);
}

template <class T>
constexpr optional<typename std::decay<T>::type> make_optional(T&& v) {
  return optional<typename std::decay<T>::type>(constexpr_forward<T>(v));
}

template <class X>
constexpr optional<X&> make_optional(std::reference_wrapper<X> v) {
  return optional<X&>(v.get());
}

} // namespace c10

namespace std {
template <typename T>
struct hash<c10::optional<T>> {
  typedef c10::invoke_result_t<std::hash<T>, T> result_type;
  typedef c10::optional<T> argument_type;

  constexpr result_type operator()(argument_type const& arg) const {
    return arg ? std::hash<T>{}(*arg) : result_type{};
  }
};

template <typename T>
struct hash<c10::optional<T&>> {
  typedef typename hash<T>::result_type result_type;
  typedef c10::optional<T&> argument_type;

  constexpr result_type operator()(argument_type const& arg) const {
    return arg ? std::hash<T>{}(*arg) : result_type{};
  }
};
} // namespace std

#undef TR2_OPTIONAL_REQUIRES
#undef TR2_OPTIONAL_ASSERTED_EXPRESSION
#undef TR2_OPTIONAL_HOST_CONSTEXPR

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

#endif // C10_UTIL_OPTIONAL_H_
