/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#if defined(__apple_build_version__) && ((__clang_major__ < 8) || ((__clang_major__ == 8) && (__clang_minor__ < 1)))

#include <emmintrin.h>

// This version of apple clang has a bug that _cvtsh_ss is not defined, see
// https://reviews.llvm.org/D16177
static __inline float
    __attribute__((__always_inline__, __nodebug__, __target__("f16c")))
_cvtsh_ss(unsigned short a)
{
  __v8hi v = {(short)a, 0, 0, 0, 0, 0, 0, 0};
  __v4sf r = __builtin_ia32_vcvtph2ps(v);
  return r[0];
}

#endif // defined(__APPLE__) && (__clang_major__ < 8)

#ifdef _MSC_VER

// It seems that microsoft msvc does not have a _cvtsh_ss implementation so
// we will add a dummy version to it.

static inline float
_cvtsh_ss(unsigned short x) {
  union {
    uint32_t intval;
    float floatval;
  } t1;
  uint32_t t2, t3;
  t1.intval = x & 0x7fff; // Non-sign bits
  t2 = x & 0x8000; // Sign bit
  t3 = x & 0x7c00; // Exponent
  t1.intval <<= 13; // Align mantissa on MSB
  t2 <<= 16; // Shift sign bit into position
  t1.intval += 0x38000000; // Adjust bias
  t1.intval = (t3 == 0 ? 0 : t1.intval); // Denormals-as-zero
  t1.intval |= t2; // Re-insert sign bit
  return t1.floatval;
}

#endif // _MSC_VER
