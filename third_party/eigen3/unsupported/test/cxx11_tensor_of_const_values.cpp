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

static void test_assign()
{
  float data1[6];
  TensorMap<Tensor<const float, 2>> mat1(data1, 2, 3);
  float data2[6];
  const TensorMap<Tensor<float, 2>> mat2(data2, 2, 3);

  for (int i = 0; i < 6; ++i) {
    data1[i] = i;
    data2[i] = -i;
  }

  Tensor<float, 2> rslt1;
  rslt1 = mat1;
  Tensor<float, 2> rslt2;
  rslt2 = mat2;

  Tensor<float, 2> rslt3 = mat1;
  Tensor<float, 2> rslt4 = mat2;

  Tensor<float, 2> rslt5(mat1);
  Tensor<float, 2> rslt6(mat2);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_APPROX(rslt1(i,j), static_cast<float>(i + 2*j));
      VERIFY_IS_APPROX(rslt2(i,j), static_cast<float>(-i - 2*j));
      VERIFY_IS_APPROX(rslt3(i,j), static_cast<float>(i + 2*j));
      VERIFY_IS_APPROX(rslt4(i,j), static_cast<float>(-i - 2*j));
      VERIFY_IS_APPROX(rslt5(i,j), static_cast<float>(i + 2*j));
      VERIFY_IS_APPROX(rslt6(i,j), static_cast<float>(-i - 2*j));
    }
  }
}


static void test_plus()
{
  float data1[6];
  TensorMap<Tensor<const float, 2>> mat1(data1, 2, 3);
  float data2[6];
  TensorMap<Tensor<float, 2>> mat2(data2, 2, 3);

  for (int i = 0; i < 6; ++i) {
    data1[i] = i;
    data2[i] = -i;
  }

  Tensor<float, 2> sum1;
  sum1 = mat1 + mat2;
  Tensor<float, 2> sum2;
  sum2 = mat2 + mat1;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_APPROX(sum1(i,j), 0.0f);
      VERIFY_IS_APPROX(sum2(i,j), 0.0f);
    }
  }
}


static void test_plus_equal()
{
  float data1[6];
  TensorMap<Tensor<const float, 2>> mat1(data1, 2, 3);
  float data2[6];
  TensorMap<Tensor<float, 2>> mat2(data2, 2, 3);

  for (int i = 0; i < 6; ++i) {
    data1[i] = i;
    data2[i] = -i;
  }
  mat2 += mat1;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_APPROX(mat2(i,j), 0.0f);
    }
  }
}


void test_cxx11_tensor_of_const_values()
{
  CALL_SUBTEST(test_assign());
  CALL_SUBTEST(test_plus());
  CALL_SUBTEST(test_plus_equal());
}
