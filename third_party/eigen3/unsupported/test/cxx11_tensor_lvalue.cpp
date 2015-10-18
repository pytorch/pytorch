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


static void test_compound_assignment()
{
  Tensor<float, 3> mat1(2,3,7);
  Tensor<float, 3> mat2(2,3,7);
  Tensor<float, 3> mat3(2,3,7);

  mat1.setRandom();
  mat2.setRandom();
  mat3 = mat1;
  mat3 += mat2;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(mat3(i,j,k), mat1(i,j,k) + mat2(i,j,k));
      }
    }
  }
}


void test_cxx11_tensor_lvalue()
{
  CALL_SUBTEST(test_compound_assignment());
}
