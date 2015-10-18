// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::TensorMap;

static void test_assign()
{
  std::string data1[6];
  TensorMap<Tensor<std::string, 2>> mat1(data1, 2, 3);
  std::string data2[6];
  const TensorMap<Tensor<const std::string, 2>> mat2(data2, 2, 3);

  for (int i = 0; i < 6; ++i) {
    std::ostringstream s1;
    s1 << "abc" << i*3;
    data1[i] = s1.str();
    std::ostringstream s2;
    s2 << "def" << i*5;
    data2[i] = s2.str();
  }

  Tensor<std::string, 2> rslt1;
  rslt1 = mat1;
  Tensor<std::string, 2> rslt2;
  rslt2 = mat2;

  Tensor<std::string, 2> rslt3 = mat1;
  Tensor<std::string, 2> rslt4 = mat2;

  Tensor<std::string, 2> rslt5(mat1);
  Tensor<std::string, 2> rslt6(mat2);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_EQUAL(rslt1(i,j), data1[i+2*j]);
      VERIFY_IS_EQUAL(rslt2(i,j), data2[i+2*j]);
      VERIFY_IS_EQUAL(rslt3(i,j), data1[i+2*j]);
      VERIFY_IS_EQUAL(rslt4(i,j), data2[i+2*j]);
      VERIFY_IS_EQUAL(rslt5(i,j), data1[i+2*j]);
      VERIFY_IS_EQUAL(rslt6(i,j), data2[i+2*j]);
    }
  }
}


static void test_concat()
{
  Tensor<std::string, 2> t1(2, 3);
  Tensor<std::string, 2> t2(2, 3);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      std::ostringstream s1;
      s1 << "abc" << i + j*2;
      t1(i, j) = s1.str();
      std::ostringstream s2;
      s2 << "def" << i*5 + j*32;
      t2(i, j) = s2.str();
    }
  }

  Tensor<std::string, 2> result = t1.concatenate(t2, 1);
  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_EQUAL(result.dimension(1), 6);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_EQUAL(result(i, j),   t1(i, j));
      VERIFY_IS_EQUAL(result(i, j+3), t2(i, j));
    }
  }
}


static void test_slices()
{
  Tensor<std::string, 2> data(2, 6);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      std::ostringstream s1;
      s1 << "abc" << i + j*2;
      data(i, j) = s1.str();
    }
  }

  const Eigen::DSizes<ptrdiff_t, 2> half_size(2, 3);
  const Eigen::DSizes<ptrdiff_t, 2> first_half(0, 0);
  const Eigen::DSizes<ptrdiff_t, 2> second_half(0, 3);

  Tensor<std::string, 2> t1 = data.slice(first_half, half_size);
  Tensor<std::string, 2> t2 = data.slice(second_half, half_size);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_EQUAL(data(i, j),   t1(i, j));
      VERIFY_IS_EQUAL(data(i, j+3), t2(i, j));
    }
  }
}


static void test_additions()
{
  Tensor<std::string, 1> data1(3);
  Tensor<std::string, 1> data2(3);
  for (int i = 0; i < 3; ++i) {
    data1(i) = "abc";
    std::ostringstream s1;
    s1 << i;
    data2(i) = s1.str();
  }

  Tensor<std::string, 1> sum = data1 + data2;
  for (int i = 0; i < 3; ++i) {
    std::ostringstream concat;
    concat << "abc" << i;
    std::string expected = concat.str();
    VERIFY_IS_EQUAL(sum(i), expected);
  }
}


static void test_initialization()
{
  Tensor<std::string, 2> a(2, 3);
  a.setConstant(std::string("foo"));
  for (int i = 0; i < 2*3; ++i) {
    VERIFY_IS_EQUAL(a(i), std::string("foo"));
  }
}


void test_cxx11_tensor_of_strings()
{
  // Beware: none of this is likely to ever work on a GPU.
  CALL_SUBTEST(test_assign());
  CALL_SUBTEST(test_concat());
  CALL_SUBTEST(test_slices());
  CALL_SUBTEST(test_additions());
  CALL_SUBTEST(test_initialization());
}
