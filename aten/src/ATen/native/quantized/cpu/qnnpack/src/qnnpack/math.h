/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#ifdef _MSC_VER
#undef min
#undef max
#endif

inline static size_t min(size_t a, size_t b) {
  return a < b ? a : b;
}

inline static size_t max(size_t a, size_t b) {
  return a > b ? a : b;
}

inline static size_t doz(size_t a, size_t b) {
  return a < b ? 0 : a - b;
}

inline static size_t divide_round_up(size_t n, size_t q) {
  return n % q == 0 ? n / q : n / q + 1;
}

inline static size_t round_up(size_t n, size_t q) {
  return divide_round_up(n, q) * q;
}
