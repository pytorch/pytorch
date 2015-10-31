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
using Eigen::RowMajor;

static void test_orderings()
{
  Tensor<float, 3> mat1(2,3,7);
  Tensor<float, 3> mat2(2,3,7);
  Tensor<bool, 3> lt(2,3,7);
  Tensor<bool, 3> le(2,3,7);
  Tensor<bool, 3> gt(2,3,7);
  Tensor<bool, 3> ge(2,3,7);

  mat1.setRandom();
  mat2.setRandom();

  lt = mat1 < mat2;
  le = mat1 <= mat2;
  gt = mat1 > mat2;
  ge = mat1 >= mat2;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(lt(i,j,k), mat1(i,j,k) < mat2(i,j,k));
        VERIFY_IS_EQUAL(le(i,j,k), mat1(i,j,k) <= mat2(i,j,k));
        VERIFY_IS_EQUAL(gt(i,j,k), mat1(i,j,k) > mat2(i,j,k));
        VERIFY_IS_EQUAL(ge(i,j,k), mat1(i,j,k) >= mat2(i,j,k));
      }
    }
  }
}


static void test_equality()
{
  Tensor<float, 3> mat1(2,3,7);
  Tensor<float, 3> mat2(2,3,7);

  mat1.setRandom();
  mat2.setRandom();
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        if (internal::random<bool>()) {
          mat2(i,j,k) = mat1(i,j,k);
        }
      }
    }
  }

  Tensor<bool, 3> eq(2,3,7);
  Tensor<bool, 3> ne(2,3,7);
  eq = (mat1 == mat2);
  ne = (mat1 != mat2);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(eq(i,j,k), mat1(i,j,k) == mat2(i,j,k));
        VERIFY_IS_EQUAL(ne(i,j,k), mat1(i,j,k) != mat2(i,j,k));
      }
    }
  }
}


void test_cxx11_tensor_comparisons()
{
  CALL_SUBTEST(test_orderings());
  CALL_SUBTEST(test_equality());
}
