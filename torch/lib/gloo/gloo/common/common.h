/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <cstdlib>
#include <malloc.h>
#include <memory>

namespace gloo {

// Align buffers to 32 bytes to support vectorized code
const size_t kBufferAlignment = 32;

template <typename T, int ALIGNMENT = kBufferAlignment>
class aligned_allocator {
 public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  struct rebind {
    using other = aligned_allocator<U, ALIGNMENT>;
  };

  inline explicit aligned_allocator() = default;
  inline ~aligned_allocator() = default;
  inline explicit aligned_allocator(const aligned_allocator& a) = default;

  inline pointer address(reference r) {
    return &r;
  }
  inline const_pointer address(const_reference r) {
    return &r;
  }

  inline pointer allocate(
      size_type sz,
      typename std::allocator<void>::const_pointer = 0) {
    auto x = memalign(ALIGNMENT, sizeof(T) * sz);
    return reinterpret_cast<pointer>(x);
  }

  void deallocate(pointer p, size_type /*sz*/) {
    free(p);
  }
};

// make_unique is a C++14 feature. If we don't have 14, we will emulate
// its behavior. This is copied from folly/Memory.h
#if __cplusplus >= 201402L ||                                              \
    (defined __cpp_lib_make_unique && __cpp_lib_make_unique >= 201304L) || \
    (defined(_MSC_VER) && _MSC_VER >= 1900)
/* using override */ using std::make_unique;
#else

template<typename T, typename... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Allows 'make_unique<T[]>(10)'. (N3690 s20.9.1.4 p3-4)
template<typename T>
typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(const size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

// Disallows 'make_unique<T[10]>()'. (N3690 s20.9.1.4 p5)
template<typename T, typename... Args>
typename std::enable_if<
  std::extent<T>::value != 0, std::unique_ptr<T>>::type
make_unique(Args&&...) = delete;

#endif

} // namespace gloo
