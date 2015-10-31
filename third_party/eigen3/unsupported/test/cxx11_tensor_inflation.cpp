// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Ke Yang <yangke@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

template<int DataLayout>
static void test_simple_inflation()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();
  array<ptrdiff_t, 4> strides;

  strides[0] = 1;
  strides[1] = 1;
  strides[2] = 1;
  strides[3] = 1;

  Tensor<float, 4, DataLayout> no_stride;
  no_stride = tensor.inflate(strides);

  VERIFY_IS_EQUAL(no_stride.dimension(0), 2);
  VERIFY_IS_EQUAL(no_stride.dimension(1), 3);
  VERIFY_IS_EQUAL(no_stride.dimension(2), 5);
  VERIFY_IS_EQUAL(no_stride.dimension(3), 7);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i,j,k,l), no_stride(i,j,k,l));
        }
      }
    }
  }

  strides[0] = 2;
  strides[1] = 4;
  strides[2] = 2;
  strides[3] = 3;
  Tensor<float, 4, DataLayout> inflated;
  inflated = tensor.inflate(strides);

  VERIFY_IS_EQUAL(inflated.dimension(0), 3);
  VERIFY_IS_EQUAL(inflated.dimension(1), 9);
  VERIFY_IS_EQUAL(inflated.dimension(2), 9);
  VERIFY_IS_EQUAL(inflated.dimension(3), 19);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 9; ++k) {
        for (int l = 0; l < 19; ++l) {
          if (i % 2 == 0 &&
              j % 4 == 0 &&
              k % 2 == 0 &&
              l % 3 == 0) {
            VERIFY_IS_EQUAL(inflated(i,j,k,l),
                            tensor(i/2, j/4, k/2, l/3));
          } else {
            VERIFY_IS_EQUAL(0, inflated(i,j,k,l));
          }
        }
      }
    }
  }
}

void test_cxx11_tensor_inflation()
{
  CALL_SUBTEST(test_simple_inflation<ColMajor>());
  CALL_SUBTEST(test_simple_inflation<RowMajor>());
}
