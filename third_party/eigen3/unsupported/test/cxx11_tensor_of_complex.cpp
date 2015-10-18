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
using Eigen::TensorMap;



static void test_additions()
{
  Tensor<std::complex<float>, 1> data1(3);
  Tensor<std::complex<float>, 1> data2(3);
  for (int i = 0; i < 3; ++i) {
    data1(i) = std::complex<float>(i, -i);
    data2(i) = std::complex<float>(i, 7 * i);
  }

  Tensor<std::complex<float>, 1> sum = data1 + data2;
  for (int i = 0; i < 3; ++i) {
    VERIFY_IS_EQUAL(sum(i),  std::complex<float>(2*i, 6*i));
  }
}


static void test_abs()
{
  Tensor<std::complex<float>, 1> data1(3);
  Tensor<std::complex<double>, 1> data2(3);
  data1.setRandom();
  data2.setRandom();

  Tensor<float, 1> abs1 = data1.abs();
  Tensor<double, 1> abs2 = data2.abs();
  for (int i = 0; i < 3; ++i) {
    VERIFY_IS_APPROX(abs1(i), std::abs(data1(i)));
    VERIFY_IS_APPROX(abs2(i), std::abs(data2(i)));
  }
}


static void test_contractions()
{
  Tensor<std::complex<float>, 4> t_left(30, 50, 8, 31);
  Tensor<std::complex<float>, 5> t_right(8, 31, 7, 20, 10);
  Tensor<std::complex<float>, 5> t_result(30, 50, 7, 20, 10);

  t_left.setRandom();
  t_right.setRandom();

  typedef Map<Matrix<std::complex<float>, Dynamic, Dynamic>> MapXcf;
  MapXcf m_left(t_left.data(), 1500, 248);
  MapXcf m_right(t_right.data(), 248, 1400);
  Matrix<std::complex<float>, Dynamic, Dynamic> m_result(1500, 1400);

  // This contraction should be equivalent to a regular matrix multiplication
  typedef Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 2> dims({{DimPair(2, 0), DimPair(3, 1)}});
  t_result = t_left.contract(t_right, dims);
  m_result = m_left * m_right;
  for (int i = 0; i < t_result.dimensions().TotalSize(); i++) {
    VERIFY_IS_APPROX(t_result.data()[i], m_result.data()[i]);
  }
}


void test_cxx11_tensor_of_complex()
{
  CALL_SUBTEST(test_additions());
  CALL_SUBTEST(test_abs());
  CALL_SUBTEST(test_contractions());
}
