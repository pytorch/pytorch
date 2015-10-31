// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <sstream>
#include <string>
#include <Eigen/CXX11/Tensor>


template<int DataLayout>
static void test_output_1d()
{
  Tensor<int, 1, DataLayout> tensor(5);
  for (int i = 0; i < 5; ++i) {
    tensor(i) = i;
  }

  std::stringstream os;
  os << tensor;

  std::string expected("0\n1\n2\n3\n4");
  VERIFY_IS_EQUAL(std::string(os.str()), expected);
}


template<int DataLayout>
static void test_output_2d()
{
  Tensor<int, 2, DataLayout> tensor(5, 3);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 3; ++j) {
      tensor(i, j) = i*j;
    }
  }

  std::stringstream os;
  os << tensor;

  std::string expected("0  0  0\n0  1  2\n0  2  4\n0  3  6\n0  4  8");
  VERIFY_IS_EQUAL(std::string(os.str()), expected);
}


template<int DataLayout>
static void test_output_expr()
{
  Tensor<int, 1, DataLayout> tensor1(5);
  Tensor<int, 1, DataLayout> tensor2(5);
  for (int i = 0; i < 5; ++i) {
    tensor1(i) = i;
    tensor2(i) = 7;
  }

  std::stringstream os;
  os << tensor1 + tensor2;

  std::string expected(" 7\n 8\n 9\n10\n11");
  VERIFY_IS_EQUAL(std::string(os.str()), expected);
}


template<int DataLayout>
static void test_output_string()
{
  Tensor<std::string, 2, DataLayout> tensor(5, 3);
  tensor.setConstant(std::string("foo"));

  std::cout << tensor << std::endl;

  std::stringstream os;
  os << tensor;

  std::string expected("foo  foo  foo\nfoo  foo  foo\nfoo  foo  foo\nfoo  foo  foo\nfoo  foo  foo");
  VERIFY_IS_EQUAL(std::string(os.str()), expected);
}


template<int DataLayout>
static void test_output_const()
{
  Tensor<int, 1, DataLayout> tensor(5);
  for (int i = 0; i < 5; ++i) {
    tensor(i) = i;
  }

  TensorMap<Tensor<const int, 1, DataLayout> > tensor_map(tensor.data(), 5);

  std::stringstream os;
  os << tensor_map;

  std::string expected("0\n1\n2\n3\n4");
  VERIFY_IS_EQUAL(std::string(os.str()), expected);
}


void test_cxx11_tensor_io()
{
  CALL_SUBTEST(test_output_1d<ColMajor>());
  CALL_SUBTEST(test_output_1d<RowMajor>());
  CALL_SUBTEST(test_output_2d<ColMajor>());
  CALL_SUBTEST(test_output_2d<RowMajor>());
  CALL_SUBTEST(test_output_expr<ColMajor>());
  CALL_SUBTEST(test_output_expr<RowMajor>());
  CALL_SUBTEST(test_output_string<ColMajor>());
  CALL_SUBTEST(test_output_string<RowMajor>());
  CALL_SUBTEST(test_output_const<ColMajor>());
  CALL_SUBTEST(test_output_const<RowMajor>());
}
