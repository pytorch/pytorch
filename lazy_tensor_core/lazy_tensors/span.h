#pragma once

#include "lazy_tensors/computation_client/debug_macros.h"

namespace lazy_tensors {

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using decay_t = typename std::decay<T>::type;

namespace span_internal {

// Wrappers for access to container data pointers.
template <typename C>
constexpr auto GetDataImpl(C& c, char) noexcept  // NOLINT(runtime/references)
    -> decltype(c.data()) {
  return c.data();
}

// Before C++17, std::string::data returns a const char* in all cases.
inline char* GetDataImpl(std::string& s,  // NOLINT(runtime/references)
                         int) noexcept {
  return &s[0];
}

template <typename C>
constexpr auto GetData(C& c) noexcept  // NOLINT(runtime/references)
    -> decltype(GetDataImpl(c, 0)) {
  return GetDataImpl(c, 0);
}

// Detection idioms for size() and data().
template <typename C>
using HasSize = std::is_integral<decay_t<decltype(std::declval<C&>().size())>>;

// We want to enable conversion from vector<T*> to Span<const T* const> but
// disable conversion from vector<Derived> to Span<Base>. Here we use
// the fact that U** is convertible to Q* const* if and only if Q is the same
// type or a more cv-qualified version of U.  We also decay the result type of
// data() to avoid problems with classes which have a member function data()
// which returns a reference.
template <typename T, typename C>
using HasData =
    std::is_convertible<decay_t<decltype(GetData(std::declval<C&>()))>*,
                        T* const*>;

}  // namespace span_internal

template <typename T>
class Span {
 private:
  // Used to determine whether a Span can be constructed from a container of
  // type C.
  template <typename C>
  using EnableIfConvertibleFrom =
      typename std::enable_if<span_internal::HasData<T, C>::value &&
                              span_internal::HasSize<C>::value>::type;

  // Used to SFINAE-enable a function when the slice elements are const.
  template <typename U>
  using EnableIfConstView =
      typename std::enable_if<std::is_const<T>::value, U>::type;

 public:
  using value_type = remove_cv_t<T>;
  using pointer = T*;
  using iterator = pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using size_type = size_t;

  constexpr Span() noexcept = default;
  constexpr Span(pointer array, size_t length) noexcept
      : ptr_(array), len_(length) {}

  // Implicit reference constructor for a read-only `Span<const T>` type
  template <typename V, typename = EnableIfConvertibleFrom<V>,
            typename = EnableIfConstView<V>>
  constexpr Span(const V& v) noexcept
      : Span(span_internal::GetData(v), v.size()) {}

  Span(std::initializer_list<value_type> v) noexcept
      : Span(v.begin(), v.size()) {}

  constexpr pointer data() const noexcept { return ptr_; }

  constexpr auto size() const noexcept { return len_; }

  constexpr bool empty() const noexcept { return size() == 0; }

  constexpr T& operator[](size_t i) const
      noexcept {  // MSVC 2015 accepts this as constexpr, but not ptr_[i]
    LTC_CHECK_LT(i, size());
    return *(data() + i);
  }

  constexpr T& at(size_t i) const {
    if (i < size()) {
      return *(data() + i);
    }
    std::out_of_range("Span::at failed bounds check");
  }

  constexpr T& front() const noexcept {
    LTC_CHECK_GT(size(), 0);
    return *data();
  }

  constexpr T& back() const noexcept {
    LTC_CHECK_GT(size(), 0);
    return *(data() + size() - 1);
  }

  constexpr auto begin() const noexcept { return data(); }

  constexpr auto end() const noexcept { return data() + size(); }

  constexpr auto rbegin() const noexcept { return reverse_iterator(end()); }

  constexpr auto rend() const noexcept { return reverse_iterator(begin()); }

  static bool equals(Span<T> a, Span<T> b) {
    static_assert(std::is_const<T>::value, "");
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
  }

 private:
  pointer ptr_;
  size_type len_;
};

template <typename T>
bool operator==(Span<T> a, Span<T> b) {
  return Span<T>::equals(a, b);
}

template <typename T>
bool operator!=(Span<T> a, Span<T> b) {
  return !(a == b);
}

// MakeSpan()
//
// Constructs a mutable `Span<T>`, deducing `T` automatically from either a
// container or pointer+size.
//
// Because a read-only `Span<const T>` is implicitly constructed from container
// types regardless of whether the container itself is a const container,
// constructing mutable spans of type `Span<T>` from containers requires
// explicit constructors. The container-accepting version of `MakeSpan()`
// deduces the type of `T` by the constness of the pointer received from the
// container's `data()` member. Similarly, the pointer-accepting version returns
// a `Span<const T>` if `T` is `const`, and a `Span<T>` otherwise.
//
// Examples:
//
//   void MyRoutine(lazy_tensors::Span<MyComplicatedType> a) {
//     ...
//   };
//   // my_vector is a container of non-const types
//   std::vector<MyComplicatedType> my_vector;
//
//   // Constructing a Span implicitly attempts to create a Span of type
//   // `Span<const T>`
//   MyRoutine(my_vector);                // error, type mismatch
//
//   // Explicitly constructing the Span is verbose
//   MyRoutine(lazy_tensors::Span<MyComplicatedType>(my_vector));
//
//   // Use MakeSpan() to make an lazy_tensors::Span<T>
//   MyRoutine(lazy_tensors::MakeSpan(my_vector));
//
//   // Construct a span from an array ptr+size
//   lazy_tensors::Span<T> my_span() {
//     return lazy_tensors::MakeSpan(&array[0], num_elements_);
//   }
//
template <int&... ExplicitArgumentBarrier, typename T>
constexpr Span<T> MakeSpan(T* ptr, size_t size) noexcept {
  return Span<T>(ptr, size);
}

template <int&... ExplicitArgumentBarrier, typename T>
Span<T> MakeSpan(T* begin, T* end) noexcept {
  LTC_CHECK_LE(begin, end);
  return Span<T>(begin, end - begin);
}

template <int&... ExplicitArgumentBarrier, typename C>
constexpr auto MakeSpan(C& c) noexcept  // NOLINT(runtime/references)
    -> decltype(lazy_tensors::MakeSpan(span_internal::GetData(c), c.size())) {
  return MakeSpan(span_internal::GetData(c), c.size());
}

template <int&... ExplicitArgumentBarrier, typename T, size_t N>
constexpr Span<T> MakeSpan(T (&array)[N]) noexcept {
  return Span<T>(array, N);
}

// MakeConstSpan()
//
// Constructs a `Span<const T>` as with `MakeSpan`, deducing `T` automatically,
// but always returning a `Span<const T>`.
//
// Examples:
//
//   void ProcessInts(lazy_tensors::Span<const int> some_ints);
//
//   // Call with a pointer and size.
//   int array[3] = { 0, 0, 0 };
//   ProcessInts(lazy_tensors::MakeConstSpan(&array[0], 3));
//
//   // Call with a [begin, end) pair.
//   ProcessInts(lazy_tensors::MakeConstSpan(&array[0], &array[3]));
//
//   // Call directly with an array.
//   ProcessInts(lazy_tensors::MakeConstSpan(array));
//
//   // Call with a contiguous container.
//   std::vector<int> some_ints = ...;
//   ProcessInts(lazy_tensors::MakeConstSpan(some_ints));
//   ProcessInts(lazy_tensors::MakeConstSpan(std::vector<int>{ 0, 0, 0 }));
//
template <int&... ExplicitArgumentBarrier, typename T>
constexpr Span<const T> MakeConstSpan(T* ptr, size_t size) noexcept {
  return Span<const T>(ptr, size);
}

template <int&... ExplicitArgumentBarrier, typename T>
Span<const T> MakeConstSpan(T* begin, T* end) noexcept {
  LTC_CHECK_LE(begin, end);
  return Span<const T>(begin, end - begin);
}

template <int&... ExplicitArgumentBarrier, typename C>
constexpr auto MakeConstSpan(const C& c) noexcept -> decltype(MakeSpan(c)) {
  return MakeSpan(c);
}

template <int&... ExplicitArgumentBarrier, typename T, size_t N>
constexpr Span<const T> MakeConstSpan(const T (&array)[N]) noexcept {
  return Span<const T>(array, N);
}

}  // namespace lazy_tensors
