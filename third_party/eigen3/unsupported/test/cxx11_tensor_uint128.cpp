// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::internal::TensorUInt128;
using Eigen::internal::static_val;

void VERIFY_EQUAL(TensorUInt128<uint64_t, uint64_t> actual, __uint128_t expected) {
  bool matchl = actual.lower() == static_cast<uint64_t>(expected);
  bool matchh = actual.upper() == static_cast<uint64_t>(expected >> 64);
  if (!matchl || !matchh) {
    const char* testname = g_test_stack.back().c_str();
    std::cerr << "Test " << testname << " failed in " << __FILE__
              << " (" << __LINE__ << ")"
              << std::endl;
    abort();
  }
}


void test_add() {
  uint64_t incr = internal::random<uint64_t>(1, 9999999999);
  for (uint64_t i1 = 0; i1 < 100; ++i1) {
    for (uint64_t i2 = 1; i2 < 100 * incr; i2 += incr) {
      TensorUInt128<uint64_t, uint64_t> i(i1, i2);
      __uint128_t a = (static_cast<__uint128_t>(i1) << 64) + static_cast<__uint128_t>(i2);
      for (uint64_t j1 = 0; j1 < 100; ++j1) {
        for (uint64_t j2 = 1; j2 < 100 * incr; j2 += incr) {
          TensorUInt128<uint64_t, uint64_t> j(j1, j2);
          __uint128_t b = (static_cast<__uint128_t>(j1) << 64) + static_cast<__uint128_t>(j2);
          TensorUInt128<uint64_t, uint64_t> actual = i + j;
          __uint128_t expected = a + b;
          VERIFY_EQUAL(actual, expected);
        }
      }
    }
  }
}

void test_sub() {
  uint64_t incr = internal::random<uint64_t>(1, 9999999999);
  for (uint64_t i1 = 0; i1 < 100; ++i1) {
    for (uint64_t i2 = 1; i2 < 100 * incr; i2 += incr) {
      TensorUInt128<uint64_t, uint64_t> i(i1, i2);
      __uint128_t a = (static_cast<__uint128_t>(i1) << 64) + static_cast<__uint128_t>(i2);
      for (uint64_t j1 = 0; j1 < 100; ++j1) {
        for (uint64_t j2 = 1; j2 < 100 * incr; j2 += incr) {
          TensorUInt128<uint64_t, uint64_t> j(j1, j2);
          __uint128_t b = (static_cast<__uint128_t>(j1) << 64) + static_cast<__uint128_t>(j2);
          TensorUInt128<uint64_t, uint64_t> actual = i - j;
          __uint128_t expected = a - b;
          VERIFY_EQUAL(actual, expected);
        }
      }
    }
  }
}

void test_mul() {
  uint64_t incr = internal::random<uint64_t>(1, 9999999999);
  for (uint64_t i1 = 0; i1 < 100; ++i1) {
    for (uint64_t i2 = 1; i2 < 100 * incr; i2 += incr) {
      TensorUInt128<uint64_t, uint64_t> i(i1, i2);
      __uint128_t a = (static_cast<__uint128_t>(i1) << 64) + static_cast<__uint128_t>(i2);
      for (uint64_t j1 = 0; j1 < 100; ++j1) {
        for (uint64_t j2 = 1; j2 < 100 * incr; j2 += incr) {
          TensorUInt128<uint64_t, uint64_t> j(j1, j2);
          __uint128_t b = (static_cast<__uint128_t>(j1) << 64) + static_cast<__uint128_t>(j2);
          TensorUInt128<uint64_t, uint64_t> actual = i * j;
          __uint128_t expected = a * b;
          VERIFY_EQUAL(actual, expected);
        }
      }
    }
  }
}

void test_div() {
  uint64_t incr = internal::random<uint64_t>(1, 9999999999);
  for (uint64_t i1 = 0; i1 < 100; ++i1) {
    for (uint64_t i2 = 1; i2 < 100 * incr; i2 += incr) {
      TensorUInt128<uint64_t, uint64_t> i(i1, i2);
      __uint128_t a = (static_cast<__uint128_t>(i1) << 64) + static_cast<__uint128_t>(i2);
      for (uint64_t j1 = 0; j1 < 100; ++j1) {
        for (uint64_t j2 = 1; j2 < 100 * incr; j2 += incr) {
          TensorUInt128<uint64_t, uint64_t> j(j1, j2);
          __uint128_t b = (static_cast<__uint128_t>(j1) << 64) + static_cast<__uint128_t>(j2);
          TensorUInt128<uint64_t, uint64_t> actual = i / j;
          __uint128_t expected = a / b;
          VERIFY_EQUAL(actual, expected);
        }
      }
    }
  }
}

void test_misc1() {
  uint64_t incr = internal::random<uint64_t>(1, 9999999999);
  for (uint64_t i2 = 1; i2 < 100 * incr; i2 += incr) {
    TensorUInt128<static_val<0>, uint64_t> i(0, i2);
    __uint128_t a = static_cast<__uint128_t>(i2);
    for (uint64_t j2 = 1; j2 < 100 * incr; j2 += incr) {
      TensorUInt128<static_val<0>, uint64_t> j(0, j2);
      __uint128_t b = static_cast<__uint128_t>(j2);
      uint64_t actual = (i * j).upper();
      uint64_t expected = (a * b) >> 64;
      VERIFY_IS_EQUAL(actual, expected);
    }
  }
}

void test_misc2() {
  int64_t incr = internal::random<int64_t>(1, 100);
  for (int64_t log_div = 0; log_div < 63; ++log_div) {
    for (int64_t divider = 1; divider <= 1000000 * incr; divider += incr) {
      uint64_t expected = (static_cast<__uint128_t>(1) << (64+log_div)) / static_cast<__uint128_t>(divider) - (static_cast<__uint128_t>(1) << 64) + 1;
      uint64_t shift = 1ULL << log_div;

      TensorUInt128<uint64_t, uint64_t> result = (TensorUInt128<uint64_t, static_val<0> >(shift, 0) / TensorUInt128<static_val<0>, uint64_t>(divider) - TensorUInt128<static_val<1>, static_val<0> >(1, 0) + TensorUInt128<static_val<0>, static_val<1> >(1));
      uint64_t actual = static_cast<uint64_t>(result);
      VERIFY_EQUAL(actual, expected);
    }
  }
}


void test_cxx11_tensor_uint128()
{
  CALL_SUBTEST_1(test_add());
  CALL_SUBTEST_2(test_sub());
  CALL_SUBTEST_3(test_mul());
  CALL_SUBTEST_4(test_div());
  CALL_SUBTEST_5(test_misc1());
  CALL_SUBTEST_6(test_misc2());
}
