#pragma once

/**
 * A constexpr std::reverse_iterator for C++11.
 * Implementation taken from libstdc++,
 * https://raw.githubusercontent.com/gcc-mirror/gcc/gcc-9_2_0-release/libstdc%2B%2B-v3/include/bits/stl_iterator.h
 * adapted to our code base and constexpr'ified.
 */

// Copyright (C) 2001-2019 Free Software Foundation, Inc.
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

/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Hewlett-Packard Company makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 *
 * Copyright (c) 1996-1998
 * Silicon Graphics Computer Systems, Inc.
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Silicon Graphics makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 */

#include <c10/util/C++17.h>
#include <iterator>

namespace c10 {

template <typename _Iterator>
class reverse_iterator
    : public std::iterator<
          typename std::iterator_traits<_Iterator>::iterator_category,
          typename std::iterator_traits<_Iterator>::value_type,
          typename std::iterator_traits<_Iterator>::difference_type,
          typename std::iterator_traits<_Iterator>::pointer,
          typename std::iterator_traits<_Iterator>::reference> {
 protected:
  _Iterator current;

  using __traits_type = std::iterator_traits<_Iterator>;

 public:
  using iterator_type = _Iterator;
  using difference_type = typename __traits_type::difference_type;
  using pointer = typename __traits_type::pointer;
  using reference = typename __traits_type::reference;

  constexpr reverse_iterator() : current() {}

  explicit constexpr reverse_iterator(iterator_type __x) : current(__x) {}

  constexpr reverse_iterator(const reverse_iterator& __x)
      : current(__x.current) {}

  constexpr reverse_iterator& operator=(
      const reverse_iterator& rhs) noexcept {
    current = rhs.current;
    return current;
  }

  template <typename _Iter>
  constexpr reverse_iterator(const reverse_iterator<_Iter>& __x)
      : current(__x.base()) {}

  constexpr iterator_type base() const {
    return current;
  }

  constexpr reference operator*() const {
#if defined(__cpp_constexpr) && __cpp_constexpr >= 201304
    _Iterator iter = current;
    return *--iter;
#else
    // Only works for random access iterators if we're not C++14 :(
    return *(current - 1);
#endif
  }

  constexpr pointer operator->() const {
#if defined(__cpp_constexpr) && __cpp_constexpr >= 201304
    _Iterator iter = current;
    return _S_to_pointer(--iter);
#else
    // Only works for random access iterators if we're not C++14 :(
    return _S_to_pointer(current - 1);
#endif
  }

  constexpr reverse_iterator& operator++() {
    --current;
    return *this;
  }

  constexpr reverse_iterator operator++(int) {
    reverse_iterator __tmp = *this;
    --current;
    return __tmp;
  }

  constexpr reverse_iterator& operator--() {
    ++current;
    return *this;
  }

  constexpr reverse_iterator operator--(int) {
    reverse_iterator __tmp = *this;
    ++current;
    return __tmp;
  }

  constexpr reverse_iterator operator+(difference_type __n) const {
    return reverse_iterator(current - __n);
  }

  constexpr reverse_iterator& operator+=(difference_type __n) {
    current -= __n;
    return *this;
  }

  constexpr reverse_iterator operator-(difference_type __n) const {
    return reverse_iterator(current + __n);
  }

  constexpr reverse_iterator& operator-=(difference_type __n) {
    current += __n;
    return *this;
  }

  constexpr reference operator[](difference_type __n) const {
    return *(*this + __n);
  }

 private:
  template <typename _Tp>
  static constexpr _Tp* _S_to_pointer(_Tp* __p) {
    return __p;
  }

  template <typename _Tp>
  static constexpr pointer _S_to_pointer(_Tp __t) {
    return __t.operator->();
  }
};

template <typename _Iterator>
inline constexpr bool operator==(
    const reverse_iterator<_Iterator>& __x,
    const reverse_iterator<_Iterator>& __y) {
  return __x.base() == __y.base();
}

template <typename _Iterator>
inline constexpr bool operator<(
    const reverse_iterator<_Iterator>& __x,
    const reverse_iterator<_Iterator>& __y) {
  return __y.base() < __x.base();
}

template <typename _Iterator>
inline constexpr bool operator!=(
    const reverse_iterator<_Iterator>& __x,
    const reverse_iterator<_Iterator>& __y) {
  return !(__x == __y);
}

template <typename _Iterator>
inline constexpr bool operator>(
    const reverse_iterator<_Iterator>& __x,
    const reverse_iterator<_Iterator>& __y) {
  return __y < __x;
}

template <typename _Iterator>
inline constexpr bool operator<=(
    const reverse_iterator<_Iterator>& __x,
    const reverse_iterator<_Iterator>& __y) {
  return !(__y < __x);
}

template <typename _Iterator>
inline constexpr bool operator>=(
    const reverse_iterator<_Iterator>& __x,
    const reverse_iterator<_Iterator>& __y) {
  return !(__x < __y);
}

template <typename _IteratorL, typename _IteratorR>
inline constexpr bool operator==(
    const reverse_iterator<_IteratorL>& __x,
    const reverse_iterator<_IteratorR>& __y) {
  return __x.base() == __y.base();
}

template <typename _IteratorL, typename _IteratorR>
inline constexpr bool operator<(
    const reverse_iterator<_IteratorL>& __x,
    const reverse_iterator<_IteratorR>& __y) {
  return __y.base() < __x.base();
}

template <typename _IteratorL, typename _IteratorR>
inline constexpr bool operator!=(
    const reverse_iterator<_IteratorL>& __x,
    const reverse_iterator<_IteratorR>& __y) {
  return !(__x == __y);
}

template <typename _IteratorL, typename _IteratorR>
inline constexpr bool operator>(
    const reverse_iterator<_IteratorL>& __x,
    const reverse_iterator<_IteratorR>& __y) {
  return __y < __x;
}

template <typename _IteratorL, typename _IteratorR>
inline constexpr bool operator<=(
    const reverse_iterator<_IteratorL>& __x,
    const reverse_iterator<_IteratorR>& __y) {
  return !(__y < __x);
}

template <typename _IteratorL, typename _IteratorR>
inline constexpr bool operator>=(
    const reverse_iterator<_IteratorL>& __x,
    const reverse_iterator<_IteratorR>& __y) {
  return !(__x < __y);
}

template <typename _IteratorL, typename _IteratorR>
inline constexpr auto operator-(
    const reverse_iterator<_IteratorL>& __x,
    const reverse_iterator<_IteratorR>& __y)
    -> decltype(__y.base() - __x.base()) {
  return __y.base() - __x.base();
}

template <typename _Iterator>
inline constexpr reverse_iterator<_Iterator> operator+(
    typename reverse_iterator<_Iterator>::difference_type __n,
    const reverse_iterator<_Iterator>& __x) {
  return reverse_iterator<_Iterator>(__x.base() - __n);
}

template <typename _Iterator>
inline constexpr reverse_iterator<_Iterator> __make_reverse_iterator(
    _Iterator __i) {
  return reverse_iterator<_Iterator>(__i);
}

template <typename _Iterator>
inline constexpr reverse_iterator<_Iterator> make_reverse_iterator(
    _Iterator __i) {
  return reverse_iterator<_Iterator>(__i);
}

template <typename _Iterator>
auto __niter_base(reverse_iterator<_Iterator> __it)
    -> decltype(__make_reverse_iterator(__niter_base(__it.base()))) {
  return __make_reverse_iterator(__niter_base(__it.base()));
}

} // namespace c10
