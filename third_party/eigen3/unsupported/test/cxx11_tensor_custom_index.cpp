// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <limits>
#include <map>

#include <Eigen/Dense>
#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;


template <int DataLayout>
static void test_map_as_index()
{
#ifdef EIGEN_HAS_SFINAE
  Tensor<float, 4, DataLayout> tensor(2, 3, 5, 7);
  tensor.setRandom();

  using NormalIndex = DSizes<ptrdiff_t, 4>;
  using CustomIndex = std::map<ptrdiff_t, ptrdiff_t>;
  CustomIndex coeffC;
  coeffC[0] = 1;
  coeffC[1] = 2;
  coeffC[2] = 4;
  coeffC[3] = 1;
  NormalIndex coeff(1,2,4,1);

  VERIFY_IS_EQUAL(tensor.coeff(coeffC), tensor.coeff(coeff));
  VERIFY_IS_EQUAL(tensor.coeffRef(coeffC), tensor.coeffRef(coeff));
#endif
}


template <int DataLayout>
static void test_matrix_as_index()
{
#ifdef EIGEN_HAS_SFINAE
  Tensor<float, 4, DataLayout> tensor(2, 3, 5, 7);
  tensor.setRandom();

  using NormalIndex = DSizes<ptrdiff_t, 4>;
  using CustomIndex = Matrix<unsigned int, 4, 1>;
  CustomIndex coeffC(1,2,4,1);
  NormalIndex coeff(1,2,4,1);

  VERIFY_IS_EQUAL(tensor.coeff(coeffC), tensor.coeff(coeff));
  VERIFY_IS_EQUAL(tensor.coeffRef(coeffC), tensor.coeffRef(coeff));
#endif
}


template <int DataLayout>
static void test_varlist_as_index()
{
#ifdef EIGEN_HAS_SFINAE
  Tensor<float, 4, DataLayout> tensor(2, 3, 5, 7);
  tensor.setRandom();

  DSizes<ptrdiff_t, 4> coeff(1,2,4,1);

  VERIFY_IS_EQUAL(tensor.coeff({1,2,4,1}), tensor.coeff(coeff));
  VERIFY_IS_EQUAL(tensor.coeffRef({1,2,4,1}), tensor.coeffRef(coeff));
#endif
}


template <int DataLayout>
static void test_sizes_as_index()
{
#ifdef EIGEN_HAS_SFINAE
  Tensor<float, 4, DataLayout> tensor(2, 3, 5, 7);
  tensor.setRandom();

  DSizes<ptrdiff_t, 4> coeff(1,2,4,1);
  Sizes<1,2,4,1> coeffC;

  VERIFY_IS_EQUAL(tensor.coeff(coeffC), tensor.coeff(coeff));
  VERIFY_IS_EQUAL(tensor.coeffRef(coeffC), tensor.coeffRef(coeff));
#endif
}


void test_cxx11_tensor_custom_index() {
  test_map_as_index<ColMajor>();
  test_map_as_index<RowMajor>();
  test_matrix_as_index<ColMajor>();
  test_matrix_as_index<RowMajor>();
  test_varlist_as_index<ColMajor>();
  test_varlist_as_index<RowMajor>();
  test_sizes_as_index<ColMajor>();
  test_sizes_as_index<RowMajor>();
}
