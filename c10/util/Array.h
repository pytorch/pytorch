/**
 * This file is based on the std::array implementation of libstdc++ at
 * https://gcc.gnu.org/onlinedocs/gcc-7.1.0/libstdc++/api/a01056_source.html
 *
 * Changes:
 *  - isolate, i.e. remove dependencies on internal libstdc++ stuff
 *  - use c++17 behavior even in c++11 or c++14
 *  - remove std::swappable special case because that doesn't work with MSVC
 *  - constexpr more things
 *  - add some features like prepend/tail
 *
 * If using std::array at runtime, feel free to either keep using std::array or
 * use this one - it doesn't really matter. For compile time computations, this
 * one here is preferred because std::array in C++11 misses some constexpr
 * specifiers, forcing these methods to be called at runtime instead of compile
 * time.
 */

// Copyright (C) 2007-2017 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.

#pragma once

#include <c10/util/C++17.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace c10 {
namespace guts {

namespace detail {
template <typename _Tp, std::size_t _Nm>
struct __array_traits final {
  using _Type = _Tp[_Nm];

  static constexpr _Tp& _S_ref(const _Type& __t, std::size_t __n) noexcept {
    return const_cast<_Tp&>(__t[__n]);
  }

  static constexpr _Tp* _S_ptr(const _Type& __t) noexcept {
    return const_cast<_Tp*>(__t);
  }
};

template <typename _Tp>
struct __array_traits<_Tp, 0> final {
  struct _Type final {};

  static constexpr _Tp& _S_ref(const _Type& __t, std::size_t) noexcept {
    return *_S_ptr(__t);
  }

  static constexpr _Tp* _S_ptr(const _Type&) noexcept {
    return nullptr;
  }
};

[[noreturn]] inline void __throw_out_of_range(std::string msg) {
  throw std::out_of_range(std::move(msg));
}
} // namespace detail

template <typename _Tp, std::size_t _Nm>
class array final {
 public:
  using value_type = _Tp;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using iterator = value_type*;
  using const_iterator = const value_type*;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

 private:
  using _AT_Type = detail::__array_traits<_Tp, _Nm>;

 public: // needs to be public member for aggregate initialization
  typename _AT_Type::_Type _M_elems;

 public:
  // No explicit construct/copy/destroy for aggregate type.

  // DR 776.
  constexpr void fill(const value_type& __u) {
    std::fill_n(begin(), size(), __u);
  }

  constexpr void swap(array& __other) {
    std::swap_ranges(begin(), end(), __other.begin());
  }

  // Iterators.
  constexpr iterator begin() noexcept {
    return iterator(data());
  }

  constexpr const_iterator begin() const noexcept {
    return const_iterator(data());
  }

  constexpr iterator end() noexcept {
    return iterator(data() + _Nm);
  }

  constexpr const_iterator end() const noexcept {
    return const_iterator(data() + _Nm);
  }

  constexpr reverse_iterator rbegin() noexcept {
    return reverse_iterator(end());
  }

  constexpr const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }

  constexpr reverse_iterator rend() noexcept {
    return reverse_iterator(begin());
  }

  constexpr const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }

  constexpr const_iterator cbegin() const noexcept {
    return const_iterator(data());
  }

  constexpr const_iterator cend() const noexcept {
    return const_iterator(data() + _Nm);
  }

  constexpr const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(end());
  }

  constexpr const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(begin());
  }

  // Capacity.
  constexpr size_type size() const noexcept {
    return _Nm;
  }

  constexpr size_type max_size() const noexcept {
    return _Nm;
  }

  constexpr bool empty() const noexcept {
    return size() == 0;
  }

  // Element access.
  constexpr reference operator[](size_type __n) noexcept {
    return _AT_Type::_S_ref(_M_elems, __n);
  }

  constexpr const_reference operator[](size_type __n) const noexcept {
    return _AT_Type::_S_ref(_M_elems, __n);
  }

  constexpr reference at(size_type __n) {
    if (__n >= _Nm) {
      detail::__throw_out_of_range(
          std::string() + "array::at: __n (which is " + to_string(__n) + ") " +
          ">= _Nm (which is " + to_string(_Nm) + ")");
    }
    return _AT_Type::_S_ref(_M_elems, __n);
  }

  constexpr const_reference at(size_type __n) const {
    // Result of conditional expression must be an lvalue so use
    // boolean ? lvalue : (throw-expr, lvalue)
    return __n < _Nm
        ? _AT_Type::_S_ref(_M_elems, __n)
        : (detail::__throw_out_of_range(
               std::string() + "array::at: __n (which is " + to_string(__n) +
               ") " + ">= _Nm (which is " + to_string(_Nm) + ")"),
           _AT_Type::_S_ref(_M_elems, 0));
  }

  constexpr reference front() noexcept {
    return *begin();
  }

  constexpr const_reference front() const noexcept {
    return _AT_Type::_S_ref(_M_elems, 0);
  }

  constexpr reference back() noexcept {
    return _Nm ? *(end() - 1) : *end();
  }

  constexpr const_reference back() const noexcept {
    return _Nm ? _AT_Type::_S_ref(_M_elems, _Nm - 1)
               : _AT_Type::_S_ref(_M_elems, 0);
  }

  constexpr pointer data() noexcept {
    return _AT_Type::_S_ptr(_M_elems);
  }

  constexpr const_pointer data() const noexcept {
    return _AT_Type::_S_ptr(_M_elems);
  }
};

#if defined(__cpp_deduction_guides) && __cpp_deduction_guides >= 201606
template <typename _Tp, typename... _Up>
array(_Tp, _Up...) -> array<
    std::enable_if_t<(std::is_same<_Tp, _Up>::value && ...), _Tp>,
    1 + sizeof...(_Up)>;
#endif

// Array comparisons.
namespace detail {
template <class T, size_t N>
constexpr inline bool array_equals_(
    const array<T, N>& lhs,
    const array<T, N>& rhs,
    size_t current_index) {
  return (current_index == N)
      ? true
      : (lhs.at(current_index) == rhs.at(current_index) &&
         array_equals_(lhs, rhs, current_index + 1));
}
template <class T, size_t N>
constexpr inline bool array_less_(
    const array<T, N>& lhs,
    const array<T, N>& rhs,
    size_t current_index) {
  return (current_index == N)
      ? false
      : (lhs.at(current_index) < rhs.at(current_index) ||
         array_less_(lhs, rhs, current_index + 1));
}
} // namespace detail
template <typename _Tp, std::size_t _Nm>
constexpr inline bool operator==(
    const array<_Tp, _Nm>& __one,
    const array<_Tp, _Nm>& __two) {
  return detail::array_equals_(__one, __two, 0);
}

template <typename _Tp, std::size_t _Nm>
constexpr inline bool operator!=(
    const array<_Tp, _Nm>& __one,
    const array<_Tp, _Nm>& __two) {
  return !(__one == __two);
}

template <typename _Tp, std::size_t _Nm>
constexpr inline bool operator<(
    const array<_Tp, _Nm>& __a,
    const array<_Tp, _Nm>& __b) {
  return detail::array_less_(__a, __b, 0);
}

template <typename _Tp, std::size_t _Nm>
constexpr inline bool operator>(
    const array<_Tp, _Nm>& __one,
    const array<_Tp, _Nm>& __two) {
  return __two < __one;
}

template <typename _Tp, std::size_t _Nm>
constexpr inline bool operator<=(
    const array<_Tp, _Nm>& __one,
    const array<_Tp, _Nm>& __two) {
  return !(__one > __two);
}

template <typename _Tp, std::size_t _Nm>
constexpr inline bool operator>=(
    const array<_Tp, _Nm>& __one,
    const array<_Tp, _Nm>& __two) {
  return !(__one < __two);
}

// Specialized algorithms.
template <typename _Tp, std::size_t _Nm>
inline void swap(array<_Tp, _Nm>& __one, array<_Tp, _Nm>& __two) noexcept(
    noexcept(__one.swap(__two))) {
  __one.swap(__two);
}

template <std::size_t _Int, typename _Tp, std::size_t _Nm>
constexpr _Tp& get(array<_Tp, _Nm>& __arr) noexcept {
  static_assert(_Int < _Nm, "array index is within bounds");
  return detail::__array_traits<_Tp, _Nm>::_S_ref(__arr._M_elems, _Int);
}

template <std::size_t _Int, typename _Tp, std::size_t _Nm>
constexpr _Tp&& get(array<_Tp, _Nm>&& __arr) noexcept {
  static_assert(_Int < _Nm, "array index is within bounds");
  return std::move(get<_Int>(__arr));
}

template <std::size_t _Int, typename _Tp, std::size_t _Nm>
constexpr const _Tp& get(const array<_Tp, _Nm>& __arr) noexcept {
  static_assert(_Int < _Nm, "array index is within bounds");
  return detail::__array_traits<_Tp, _Nm>::_S_ref(__arr._M_elems, _Int);
}

/**
 * Some added features not available in std::array.
 * Only call these at compile time, they're slow if called at runtime.
 * Examples:
 *  tail({2, 3, 4}) == {3, 4}
 *  prepend(2, {3, 4}) == {2, 3, 4}
 */
namespace detail {
template <class T, size_t N, size_t... INDEX>
constexpr inline array<T, N - 1> tail_(
    const array<T, N>& arg,
    std::index_sequence<INDEX...>) {
  static_assert(sizeof...(INDEX) == N - 1, "invariant");
  return {{get<INDEX + 1>(arg)...}};
}
} // namespace detail
template <class T, size_t N>
constexpr inline array<T, N - 1> tail(const array<T, N>& arg) {
  static_assert(
      N > 0, "Can only call tail() on an array with at least one element");
  return detail::tail_(arg, std::make_index_sequence<N - 1>());
}

namespace detail {
template <class T, size_t N, size_t... INDEX>
constexpr inline array<T, N + 1> prepend_(
    T&& head,
    const array<T, N>& tail,
    std::index_sequence<INDEX...>) {
  return {{std::forward<T>(head), get<INDEX>(tail)...}};
}
} // namespace detail
template <class T, size_t N>
constexpr inline array<T, N + 1> prepend(T&& head, const array<T, N>& tail) {
  return detail::prepend_(
      std::forward<T>(head), tail, std::make_index_sequence<N>());
}

/**
 * Convert a C array into a std::array.
 * Example:
 *   int source[3] = {2, 3, 4};
 *   std::array<int, 3> target = to_std_array(source);
 */

namespace detail {
template <class T, size_t N, size_t... INDEX>
constexpr array<T, N> to_array_(
    const T (&arr)[N],
    std::index_sequence<INDEX...>) {
  return {{arr[INDEX]...}};
}
} // namespace detail

template <class T, size_t N>
constexpr array<T, N> to_array(const T (&arr)[N]) {
  return detail::to_array_(arr, std::make_index_sequence<N>());
}

} // namespace guts
} // namespace c10
