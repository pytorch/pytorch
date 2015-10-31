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


static void test_simple_assign()
{
  Tensor<int, 3> random(2,3,7);
  random.setRandom();

  TensorMap<Tensor<const int, 3> > constant(random.data(), 2, 3, 7);
  Tensor<int, 3> result(2,3,7);
  result = constant;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL((result(i,j,k)), random(i,j,k));
      }
    }
  }
}


static void test_assign_of_const_tensor()
{
  Tensor<int, 3> random(2,3,7);
  random.setRandom();

  TensorMap<Tensor<const int, 3> > constant1(random.data(), 2, 3, 7);
  TensorMap<const Tensor<int, 3> > constant2(random.data(), 2, 3, 7);
  const TensorMap<Tensor<int, 3> > constant3(random.data(), 2, 3, 7);

  Tensor<int, 2> result1 = constant1.chip(0, 2);
  Tensor<int, 2> result2 = constant2.chip(0, 2);
  Tensor<int, 2> result3 = constant3.chip(0, 2);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_EQUAL((result1(i,j)), random(i,j,0));
      VERIFY_IS_EQUAL((result2(i,j)), random(i,j,0));
      VERIFY_IS_EQUAL((result3(i,j)), random(i,j,0));
    }
  }
}


void test_cxx11_tensor_const()
{
  CALL_SUBTEST(test_simple_assign());
  CALL_SUBTEST(test_assign_of_const_tensor());
}
