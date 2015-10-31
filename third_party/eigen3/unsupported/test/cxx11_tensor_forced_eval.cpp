// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>

using Eigen::MatrixXf;
using Eigen::Tensor;

static void test_simple()
{
  MatrixXf m1(3,3);
  MatrixXf m2(3,3);
  m1.setRandom();
  m2.setRandom();

  TensorMap<Tensor<float, 2>> mat1(m1.data(), 3,3);
  TensorMap<Tensor<float, 2>> mat2(m2.data(), 3,3);

  Tensor<float, 2> mat3(3,3);
  mat3 = mat1;

  typedef Tensor<float, 1>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims({{DimPair(1, 0)}});

  mat3 = mat3.contract(mat2, dims).eval();

  VERIFY_IS_APPROX(mat3(0, 0), (m1*m2).eval()(0,0));
  VERIFY_IS_APPROX(mat3(0, 1), (m1*m2).eval()(0,1));
  VERIFY_IS_APPROX(mat3(0, 2), (m1*m2).eval()(0,2));
  VERIFY_IS_APPROX(mat3(1, 0), (m1*m2).eval()(1,0));
  VERIFY_IS_APPROX(mat3(1, 1), (m1*m2).eval()(1,1));
  VERIFY_IS_APPROX(mat3(1, 2), (m1*m2).eval()(1,2));
  VERIFY_IS_APPROX(mat3(2, 0), (m1*m2).eval()(2,0));
  VERIFY_IS_APPROX(mat3(2, 1), (m1*m2).eval()(2,1));
  VERIFY_IS_APPROX(mat3(2, 2), (m1*m2).eval()(2,2));
}


static void test_const()
{
  MatrixXf input(3,3);
  input.setRandom();
  MatrixXf output = input;
  output.rowwise() -= input.colwise().maxCoeff();

  Eigen::array<int, 1> depth_dim;
  depth_dim[0] = 0;
  Tensor<float, 2>::Dimensions dims2d;
  dims2d[0] = 1;
  dims2d[1] = 3;
  Eigen::array<int, 2> bcast;
  bcast[0] = 3;
  bcast[1] = 1;
  const TensorMap<Tensor<const float, 2>> input_tensor(input.data(), 3, 3);
  Tensor<float, 2> output_tensor= (input_tensor - input_tensor.maximum(depth_dim).eval().reshape(dims2d).broadcast(bcast));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_APPROX(output(i, j), output_tensor(i, j));
    }
  }
}


void test_cxx11_tensor_forced_eval()
{
  CALL_SUBTEST(test_simple());
  CALL_SUBTEST(test_const());
}
