/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <type_traits>

template <typename T, size_t Alignment>
class AlignedAllocator;

template <size_t Alignment>
class AlignedAllocator<void, Alignment> {
 public:
  typedef void* pointer;
  typedef const void* const_pointer;
  typedef void value_type;

  template <class U>
  struct rebind {
    typedef AlignedAllocator<U, Alignment> other;
  };
};

template <typename T, size_t Alignment>
class AlignedAllocator {
 public:
  typedef T value_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

#if __cplusplus >= 201402L
  typedef std::true_type propagate_on_container_move_assignment;
#endif

  template <class U>
  struct rebind {
    typedef AlignedAllocator<U, Alignment> other;
  };

 public:
  inline AlignedAllocator() noexcept = default;

  template <class U>
  inline AlignedAllocator(
      const AlignedAllocator<U, Alignment>& other) noexcept {}

  inline size_type max_size() const noexcept {
    return (std::numeric_limits<size_type>::max() - size_type(Alignment)) /
        sizeof(T);
  }

  inline pointer address(reference x) const noexcept {
    return std::addressof(x);
  }

  inline const_pointer address(const_reference x) const noexcept {
    return std::addressof(x);
  }

  inline pointer allocate(
      size_type n,
      typename AlignedAllocator<void, Alignment>::const_pointer hint = 0) {
    void* memory = std::aligned_alloc(Alignment, n * sizeof(T));
    if (!memory) {
      throw std::bad_alloc();
    }
    return static_cast<pointer>(memory);
  }

  inline void deallocate(pointer p, size_type n) noexcept {
    free(static_cast<void*>(p));
  }

  template <class U, class... Args>
  inline void construct(U* p, Args&&... args) {
    ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
  }

  template <class U>
  inline void destroy(U* p) {
    p->~U();
  }
};
