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


static void test_simple()
{
  Tensor<float, 1, ColMajor> vec1({6});
  Tensor<float, 1, ColMajor, int> vec2({6});

  vec1(0) = 4.0;  vec2(0) = 0.0;
  vec1(1) = 8.0;  vec2(1) = 1.0;
  vec1(2) = 15.0; vec2(2) = 2.0;
  vec1(3) = 16.0; vec2(3) = 3.0;
  vec1(4) = 23.0; vec2(4) = 4.0;
  vec1(5) = 42.0; vec2(5) = 5.0;

  float data3[6];
  TensorMap<Tensor<float, 1, ColMajor>> vec3(data3, 6);
  vec3 = vec1.sqrt();
  float data4[6];
  TensorMap<Tensor<float, 1, ColMajor, int>> vec4(data4, 6);
  vec4 = vec2.square();

  VERIFY_IS_APPROX(vec3(0), sqrtf(4.0));
  VERIFY_IS_APPROX(vec3(1), sqrtf(8.0));
  VERIFY_IS_APPROX(vec3(2), sqrtf(15.0));
  VERIFY_IS_APPROX(vec3(3), sqrtf(16.0));
  VERIFY_IS_APPROX(vec3(4), sqrtf(23.0));
  VERIFY_IS_APPROX(vec3(5), sqrtf(42.0));

  VERIFY_IS_APPROX(vec4(0), 0.0f);
  VERIFY_IS_APPROX(vec4(1), 1.0f);
  VERIFY_IS_APPROX(vec4(2), 2.0f * 2.0f);
  VERIFY_IS_APPROX(vec4(3), 3.0f * 3.0f);
  VERIFY_IS_APPROX(vec4(4), 4.0f * 4.0f);
  VERIFY_IS_APPROX(vec4(5), 5.0f * 5.0f);
}


void test_cxx11_tensor_mixed_indices()
{
  CALL_SUBTEST(test_simple());
}
