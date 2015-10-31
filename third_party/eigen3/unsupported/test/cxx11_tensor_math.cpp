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

using Eigen::Tensor;
using Eigen::RowMajor;

static void test_tanh()
{
  Tensor<float, 1> vec1({6});
  vec1.setRandom();

  Tensor<float, 1> vec2 = vec1.tanh();

  for (int i = 0; i < 6; ++i) {
    VERIFY_IS_APPROX(vec2(i), tanhf(vec1(i)));
  }
}

static void test_sigmoid()
{
  Tensor<float, 1> vec1({6});
  vec1.setRandom();

  Tensor<float, 1> vec2 = vec1.sigmoid();

  for (int i = 0; i < 6; ++i) {
    VERIFY_IS_APPROX(vec2(i), 1.0f / (1.0f + std::exp(-vec1(i))));
  }
}


void test_cxx11_tensor_math()
{
  CALL_SUBTEST(test_tanh());
  CALL_SUBTEST(test_sigmoid());
}
