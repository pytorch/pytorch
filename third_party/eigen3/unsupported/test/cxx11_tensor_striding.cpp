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

template<int DataLayout>
static void test_simple_striding()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();
  array<ptrdiff_t, 4> strides;
  strides[0] = 1;
  strides[1] = 1;
  strides[2] = 1;
  strides[3] = 1;

  Tensor<float, 4, DataLayout> no_stride;
  no_stride = tensor.stride(strides);

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
  Tensor<float, 4, DataLayout> stride;
  stride = tensor.stride(strides);

  VERIFY_IS_EQUAL(stride.dimension(0), 1);
  VERIFY_IS_EQUAL(stride.dimension(1), 1);
  VERIFY_IS_EQUAL(stride.dimension(2), 3);
  VERIFY_IS_EQUAL(stride.dimension(3), 3);

  for (int i = 0; i < 1; ++i) {
    for (int j = 0; j < 1; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          VERIFY_IS_EQUAL(tensor(2*i,4*j,2*k,3*l), stride(i,j,k,l));
        }
      }
    }
  }
}


template<int DataLayout>
static void test_striding_as_lvalue()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();
  array<ptrdiff_t, 4> strides;
  strides[0] = 2;
  strides[1] = 4;
  strides[2] = 2;
  strides[3] = 3;

  Tensor<float, 4, DataLayout> result(3, 12, 10, 21);
  result.stride(strides) = tensor;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i,j,k,l), result(2*i,4*j,2*k,3*l));
        }
      }
    }
  }

  array<ptrdiff_t, 4> no_strides;
  no_strides[0] = 1;
  no_strides[1] = 1;
  no_strides[2] = 1;
  no_strides[3] = 1;
  Tensor<float, 4, DataLayout> result2(3, 12, 10, 21);
  result2.stride(strides) = tensor.stride(no_strides);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i,j,k,l), result2(2*i,4*j,2*k,3*l));
        }
      }
    }
  }
}


void test_cxx11_tensor_striding()
{
  CALL_SUBTEST(test_simple_striding<ColMajor>());
  CALL_SUBTEST(test_simple_striding<RowMajor>());
  CALL_SUBTEST(test_striding_as_lvalue<ColMajor>());
  CALL_SUBTEST(test_striding_as_lvalue<RowMajor>());
}
