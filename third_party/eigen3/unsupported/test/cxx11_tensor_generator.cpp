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

struct Generator1D {
  Generator1D() { }

  float operator()(const array<Eigen::DenseIndex, 1>& coordinates) const {
    return coordinates[0];
  }
};

template <int DataLayout>
static void test_1D()
{
  Tensor<float, 1> vec(6);
  Tensor<float, 1> result = vec.generate(Generator1D());

  for (int i = 0; i < 6; ++i) {
    VERIFY_IS_EQUAL(result(i), i);
  }
}


struct Generator2D {
  Generator2D() { }

  float operator()(const array<Eigen::DenseIndex, 2>& coordinates) const {
    return 3 * coordinates[0] + 11 * coordinates[1];
  }
};

template <int DataLayout>
static void test_2D()
{
  Tensor<float, 2> matrix(5, 7);
  Tensor<float, 2> result = matrix.generate(Generator2D());

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      VERIFY_IS_EQUAL(result(i, j), 3*i + 11*j);
    }
  }
}


template <int DataLayout>
static void test_gaussian()
{
  int rows = 32;
  int cols = 48;
  array<float, 2> means;
  means[0] = rows / 2.0f;
  means[1] = cols / 2.0f;
  array<float, 2> std_devs;
  std_devs[0] = 3.14f;
  std_devs[1] = 2.7f;
  internal::GaussianGenerator<float, Eigen::DenseIndex, 2> gaussian_gen(means, std_devs);

  Tensor<float, 2> matrix(rows, cols);
  Tensor<float, 2> result = matrix.generate(gaussian_gen);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      float g_rows = powf(rows/2.0f - i, 2) / (3.14f * 3.14f) * 0.5f;
      float g_cols = powf(cols/2.0f - j, 2) / (2.7f * 2.7f) * 0.5f;
      float gaussian = expf(-g_rows - g_cols);
      VERIFY_IS_EQUAL(result(i, j), gaussian);
    }
  }
}


void test_cxx11_tensor_generator()
{
  CALL_SUBTEST(test_1D<ColMajor>());
  CALL_SUBTEST(test_1D<RowMajor>());
  CALL_SUBTEST(test_2D<ColMajor>());
  CALL_SUBTEST(test_2D<RowMajor>());
  CALL_SUBTEST(test_gaussian<ColMajor>());
  CALL_SUBTEST(test_gaussian<RowMajor>());
}
