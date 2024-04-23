// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Try to use either `std::span` or `gsl::span`
#if defined(OPENTELEMETRY_STL_VERSION)
#  if OPENTELEMETRY_STL_VERSION >= 2020
#    include <array>
#    include <cstddef>
#    include <iterator>
#    include <type_traits>

/**
 * @brief Clang 14.0.0 with libc++ do not support implicitly construct a span
 * for a range. We just use our fallback version.
 *
 */
#    if !defined(OPENTELEMETRY_OPTION_USE_STD_SPAN) && defined(_LIBCPP_VERSION)
#      if _LIBCPP_VERSION <= 14000
#        define OPENTELEMETRY_OPTION_USE_STD_SPAN 0
#      endif
#    endif
#    ifndef OPENTELEMETRY_OPTION_USE_STD_SPAN
#      define OPENTELEMETRY_OPTION_USE_STD_SPAN 1
#    endif
#    if OPENTELEMETRY_OPTION_USE_STD_SPAN
#      include "opentelemetry/std/span.h"
#    endif
#  endif /* OPENTELEMETRY_STL_VERSION >= 2020 */
#endif   /* OPENTELEMETRY_STL_VERSION */

// Fallback to `nostd::span` if necessary
#if !defined(OPENTELEMETRY_HAVE_SPAN)
#  include <array>
#  include <cassert>
#  include <cstddef>
#  include <exception>
#  include <iterator>
#  include <type_traits>

#  include "opentelemetry/nostd/utility.h"
#  include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace nostd
{
constexpr size_t dynamic_extent = static_cast<size_t>(-1);

template <class T, size_t Extent = dynamic_extent>
class span;

namespace detail
{
/**
 * Helper class to resolve overloaded constructors
 */
template <class T>
struct is_specialized_span_convertible : std::false_type
{};

template <class T, size_t N>
struct is_specialized_span_convertible<std::array<T, N>> : std::true_type
{};

template <class T, size_t N>
struct is_specialized_span_convertible<T[N]> : std::true_type
{};

template <class T, size_t Extent>
struct is_specialized_span_convertible<span<T, Extent>> : std::true_type
{};
}  // namespace detail

/**
 * Back port of std::span.
 *
 * See https://en.cppreference.com/w/cpp/container/span for interface documentation.
 *
 * Note: This provides a subset of the methods available on std::span.
 *
 * Note: The std::span API specifies error cases to have undefined behavior, so this implementation
 * chooses to terminate or assert rather than throw exceptions.
 */
template <class T, size_t Extent>
class span
{
public:
  static constexpr size_t extent = Extent;

  // This arcane code is how we make default-construction result in an SFINAE error
  // with C++11 when Extent != 0 as specified by the std::span API.
  //
  // See https://stackoverflow.com/a/10309720/4447365
  template <bool B = Extent == 0, typename std::enable_if<B>::type * = nullptr>
  span() noexcept : data_{nullptr}
  {}

  span(T *data, size_t count) noexcept : data_{data}
  {
    if (count != Extent)
    {
      std::terminate();
    }
  }

  span(T *first, T *last) noexcept : data_{first}
  {
    if (std::distance(first, last) != Extent)
    {
      std::terminate();
    }
  }

  template <size_t N, typename std::enable_if<Extent == N>::type * = nullptr>
  span(T (&array)[N]) noexcept : data_{array}
  {}

  template <size_t N, typename std::enable_if<Extent == N>::type * = nullptr>
  span(std::array<T, N> &array) noexcept : data_{array.data()}
  {}

  template <size_t N, typename std::enable_if<Extent == N>::type * = nullptr>
  span(const std::array<T, N> &array) noexcept : data_{array.data()}
  {}

  template <
      class C,
      typename std::enable_if<!detail::is_specialized_span_convertible<C>::value &&
                              std::is_convertible<typename std::remove_pointer<decltype(nostd::data(
                                                      std::declval<C &>()))>::type (*)[],
                                                  T (*)[]>::value &&
                              std::is_convertible<decltype(nostd::size(std::declval<const C &>())),
                                                  size_t>::value>::type * = nullptr>
  span(C &c) noexcept(noexcept(nostd::data(c), nostd::size(c))) : data_{nostd::data(c)}
  {
    if (nostd::size(c) != Extent)
    {
      std::terminate();
    }
  }

  template <
      class C,
      typename std::enable_if<!detail::is_specialized_span_convertible<C>::value &&
                              std::is_convertible<typename std::remove_pointer<decltype(nostd::data(
                                                      std::declval<const C &>()))>::type (*)[],
                                                  T (*)[]>::value &&
                              std::is_convertible<decltype(nostd::size(std::declval<const C &>())),
                                                  size_t>::value>::type * = nullptr>
  span(const C &c) noexcept(noexcept(nostd::data(c), nostd::size(c))) : data_{nostd::data(c)}
  {
    if (nostd::size(c) != Extent)
    {
      std::terminate();
    }
  }

  template <class U,
            size_t N,
            typename std::enable_if<N == Extent &&
                                    std::is_convertible<U (*)[], T (*)[]>::value>::type * = nullptr>
  span(const span<U, N> &other) noexcept : data_{other.data()}
  {}

  span(const span &) noexcept = default;

  span &operator=(const span &) noexcept = default;

  bool empty() const noexcept { return Extent == 0; }

  T *data() const noexcept { return data_; }

  size_t size() const noexcept { return Extent; }

  T &operator[](size_t index) const noexcept
  {
    assert(index < Extent);
    return data_[index];
  }

  T *begin() const noexcept { return data_; }

  T *end() const noexcept { return data_ + Extent; }

private:
  T *data_;
};

template <class T>
class span<T, dynamic_extent>
{
public:
  static constexpr size_t extent = dynamic_extent;

  span() noexcept : extent_{0}, data_{nullptr} {}

  span(T *data, size_t count) noexcept : extent_{count}, data_{data} {}

  span(T *first, T *last) noexcept
      : extent_{static_cast<size_t>(std::distance(first, last))}, data_{first}
  {
    assert(first <= last);
  }

  template <size_t N>
  span(T (&array)[N]) noexcept : extent_{N}, data_{array}
  {}

  template <size_t N>
  span(std::array<T, N> &array) noexcept : extent_{N}, data_{array.data()}
  {}

  template <size_t N>
  span(const std::array<T, N> &array) noexcept : extent_{N}, data_{array.data()}
  {}

  template <
      class C,
      typename std::enable_if<!detail::is_specialized_span_convertible<C>::value &&
                              std::is_convertible<typename std::remove_pointer<decltype(nostd::data(
                                                      std::declval<C &>()))>::type (*)[],
                                                  T (*)[]>::value &&
                              std::is_convertible<decltype(nostd::size(std::declval<const C &>())),
                                                  size_t>::value>::type * = nullptr>
  span(C &c) noexcept(noexcept(nostd::data(c), nostd::size(c)))
      : extent_{nostd::size(c)}, data_{nostd::data(c)}
  {}

  template <
      class C,
      typename std::enable_if<!detail::is_specialized_span_convertible<C>::value &&
                              std::is_convertible<typename std::remove_pointer<decltype(nostd::data(
                                                      std::declval<const C &>()))>::type (*)[],
                                                  T (*)[]>::value &&
                              std::is_convertible<decltype(nostd::size(std::declval<const C &>())),
                                                  size_t>::value>::type * = nullptr>
  span(const C &c) noexcept(noexcept(nostd::data(c), nostd::size(c)))
      : extent_{nostd::size(c)}, data_{nostd::data(c)}
  {}

  template <class U,
            size_t N,
            typename std::enable_if<std::is_convertible<U (*)[], T (*)[]>::value>::type * = nullptr>
  span(const span<U, N> &other) noexcept : extent_{other.size()}, data_{other.data()}
  {}

  span(const span &) noexcept = default;

  span &operator=(const span &) noexcept = default;

  bool empty() const noexcept { return extent_ == 0; }

  T *data() const noexcept { return data_; }

  size_t size() const noexcept { return extent_; }

  T &operator[](size_t index) const noexcept
  {
    assert(index < extent_);
    return data_[index];
  }

  T *begin() const noexcept { return data_; }

  T *end() const noexcept { return data_ + extent_; }

private:
  size_t extent_;
  T *data_;
};
}  // namespace nostd
OPENTELEMETRY_END_NAMESPACE
#endif
