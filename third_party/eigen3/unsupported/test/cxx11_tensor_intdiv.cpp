// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014-2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>


void test_signed_32bit()
{
  // Divide by one
  const Eigen::internal::TensorIntDivisor<int32_t, false> div_by_one(1);

  for (int32_t j = 0; j < 25000; ++j) {
    const int32_t fast_div = j / div_by_one;
    const int32_t slow_div = j / 1;
    VERIFY_IS_EQUAL(fast_div, slow_div);
  }

  // Standard divide by 2 or more
  for (int32_t i = 2; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<int32_t, false> div(i);

    for (int32_t j = 0; j < 25000; ++j) {
      const int32_t fast_div = j / div;
      const int32_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }

  // Optimized divide by 2 or more
  for (int32_t i = 2; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<int32_t, true> div(i);

    for (int32_t j = 0; j < 25000; ++j) {
      const int32_t fast_div = j / div;
      const int32_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }
}


void test_unsigned_32bit()
{
  for (uint32_t i = 1; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<uint32_t> div(i);

    for (uint32_t j = 0; j < 25000; ++j) {
      const uint32_t fast_div = j / div;
      const uint32_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }
}


void test_signed_64bit()
{
  for (int64_t i = 1; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<int64_t> div(i);

    for (int64_t j = 0; j < 25000; ++j) {
      const int64_t fast_div = j / div;
      const int64_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }
}


void test_unsigned_64bit()
{
  for (uint64_t i = 1; i < 25000; ++i) {
    const Eigen::internal::TensorIntDivisor<uint64_t> div(i);

    for (uint64_t j = 0; j < 25000; ++j) {
      const uint64_t fast_div = j / div;
      const uint64_t slow_div = j / i;
      VERIFY_IS_EQUAL(fast_div, slow_div);
    }
  }
}

void test_powers_32bit() {
  for (int expon = 1; expon < 31; expon++) {
    int32_t div = (1 << expon);
    for (int num_expon = 0; num_expon < 32; num_expon++) {
      int32_t start_num = (1 << num_expon) - 100;
      int32_t end_num = (1 << num_expon) + 100;
      if (start_num < 0)
        start_num = 0;
      for (int32_t num = start_num; num < end_num; num++) {
        Eigen::internal::TensorIntDivisor<int32_t> divider =
          Eigen::internal::TensorIntDivisor<int32_t>(div);
        int32_t result = num/div;
        int32_t result_op = divider.divide(num);
        VERIFY_IS_EQUAL(result_op, result);
      }
    }
  }
}

void test_powers_64bit() {
  for (int expon = 0; expon < 63; expon++) {
    int64_t div = (1ull << expon);
    for (int num_expon = 0; num_expon < 63; num_expon++) {
      int64_t start_num = (1ull << num_expon) - 10;
      int64_t end_num = (1ull << num_expon) + 10;
      if (start_num < 0)
        start_num = 0;
      for (int64_t num = start_num; num < end_num; num++) {
        Eigen::internal::TensorIntDivisor<int64_t> divider(div);
        int64_t result = num/div;
        int64_t result_op = divider.divide(num);
        VERIFY_IS_EQUAL(result_op, result);
      }
    }
  }
}

void test_specific() {
  // A particular combination that was previously failing
  int64_t div = 209715200;
  int64_t num = 3238002688;
  Eigen::internal::TensorIntDivisor<int64_t> divider(div);
  int64_t result = num/div;
  int64_t result_op = divider.divide(num);
  VERIFY_IS_EQUAL(result, result_op);
}

void test_cxx11_tensor_intdiv()
{
  CALL_SUBTEST_1(test_signed_32bit());
  CALL_SUBTEST_2(test_unsigned_32bit());
  CALL_SUBTEST_3(test_signed_64bit());
  CALL_SUBTEST_4(test_unsigned_64bit());
  CALL_SUBTEST_5(test_powers_32bit());
  CALL_SUBTEST_6(test_powers_64bit());
  CALL_SUBTEST_7(test_specific());
}
