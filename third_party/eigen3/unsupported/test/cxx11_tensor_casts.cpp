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
using Eigen::array;

static void test_simple_cast()
{
  Tensor<float, 2> ftensor(20,30);
  ftensor = ftensor.random() * 100.f;
  Tensor<char, 2> chartensor(20,30);
  chartensor.setRandom();
  Tensor<std::complex<float>, 2> cplextensor(20,30);
  cplextensor.setRandom();

  chartensor = ftensor.cast<char>();
  cplextensor = ftensor.cast<std::complex<float> >();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_EQUAL(chartensor(i,j), static_cast<char>(ftensor(i,j)));
      VERIFY_IS_EQUAL(cplextensor(i,j), static_cast<std::complex<float> >(ftensor(i,j)));
    }
  }
}


static void test_vectorized_cast()
{
  Tensor<int, 2> itensor(20,30);
  itensor = itensor.random() / 1000;
  Tensor<float, 2> ftensor(20,30);
  ftensor.setRandom();
  Tensor<double, 2> dtensor(20,30);
  dtensor.setRandom();

  ftensor = itensor.cast<float>();
  dtensor = itensor.cast<double>();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_EQUAL(itensor(i,j), static_cast<int>(ftensor(i,j)));
      VERIFY_IS_EQUAL(dtensor(i,j), static_cast<double>(ftensor(i,j)));
    }
  }
}


static void test_float_to_int_cast()
{
  Tensor<float, 2> ftensor(20,30);
  ftensor = ftensor.random() * 1000.0f;
  Tensor<double, 2> dtensor(20,30);
  dtensor = dtensor.random() * 1000.0;

  Tensor<int, 2> i1tensor = ftensor.cast<int>();
  Tensor<int, 2> i2tensor = dtensor.cast<int>();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_EQUAL(i1tensor(i,j), static_cast<int>(ftensor(i,j)));
      VERIFY_IS_EQUAL(i2tensor(i,j), static_cast<int>(dtensor(i,j)));
    }
  }
}


static void test_big_to_small_type_cast()
{
  Tensor<double, 2> dtensor(20, 30);
  dtensor.setRandom();
  Tensor<float, 2> ftensor(20, 30);
  ftensor = dtensor.cast<float>();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_APPROX(dtensor(i,j), static_cast<double>(ftensor(i,j)));
    }
  }
}


static void test_small_to_big_type_cast()
{
  Tensor<float, 2> ftensor(20, 30);
  ftensor.setRandom();
  Tensor<double, 2> dtensor(20, 30);
  dtensor = ftensor.cast<double>();

  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 30; ++j) {
      VERIFY_IS_APPROX(dtensor(i,j), static_cast<double>(ftensor(i,j)));
    }
  }
}


void test_cxx11_tensor_casts()
{
   CALL_SUBTEST(test_simple_cast());
   CALL_SUBTEST(test_vectorized_cast());
   CALL_SUBTEST(test_float_to_int_cast());
   CALL_SUBTEST(test_big_to_small_type_cast());
   CALL_SUBTEST(test_small_to_big_type_cast());
}
