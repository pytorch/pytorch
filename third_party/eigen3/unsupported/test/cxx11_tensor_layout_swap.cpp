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

static void test_simple_swap()
{
  Tensor<float, 3, ColMajor> tensor(2,3,7);
  tensor.setRandom();

  Tensor<float, 3, RowMajor> tensor2 = tensor.swap_layout();
  VERIFY_IS_EQUAL(tensor.dimension(0), tensor2.dimension(2));
  VERIFY_IS_EQUAL(tensor.dimension(1), tensor2.dimension(1));
  VERIFY_IS_EQUAL(tensor.dimension(2), tensor2.dimension(0));

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(tensor(i,j,k), tensor2(k,j,i));
      }
    }
  }
}


static void test_swap_as_lvalue()
{
  Tensor<float, 3, ColMajor> tensor(2,3,7);
  tensor.setRandom();

  Tensor<float, 3, RowMajor> tensor2(7,3,2);
  tensor2.swap_layout() = tensor;
  VERIFY_IS_EQUAL(tensor.dimension(0), tensor2.dimension(2));
  VERIFY_IS_EQUAL(tensor.dimension(1), tensor2.dimension(1));
  VERIFY_IS_EQUAL(tensor.dimension(2), tensor2.dimension(0));

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(tensor(i,j,k), tensor2(k,j,i));
      }
    }
  }
}


void test_cxx11_tensor_layout_swap()
{
  CALL_SUBTEST(test_simple_swap());
  CALL_SUBTEST(test_swap_as_lvalue());
}
