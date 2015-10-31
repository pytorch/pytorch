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
static void test_simple_padding()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();

  array<std::pair<ptrdiff_t, ptrdiff_t>, 4> paddings;
  paddings[0] = std::make_pair(0, 0);
  paddings[1] = std::make_pair(2, 1);
  paddings[2] = std::make_pair(3, 4);
  paddings[3] = std::make_pair(0, 0);

  Tensor<float, 4, DataLayout> padded;
  padded = tensor.pad(paddings);

  VERIFY_IS_EQUAL(padded.dimension(0), 2+0);
  VERIFY_IS_EQUAL(padded.dimension(1), 3+3);
  VERIFY_IS_EQUAL(padded.dimension(2), 5+7);
  VERIFY_IS_EQUAL(padded.dimension(3), 7+0);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 6; ++j) {
      for (int k = 0; k < 12; ++k) {
        for (int l = 0; l < 7; ++l) {
          if (j >= 2 && j < 5 && k >= 3 && k < 8) {
            VERIFY_IS_EQUAL(padded(i,j,k,l), tensor(i,j-2,k-3,l));
          } else {
            VERIFY_IS_EQUAL(padded(i,j,k,l), 0.0f);
          }
        }
      }
    }
  }
}

template<int DataLayout>
static void test_padded_expr()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();

  array<std::pair<ptrdiff_t, ptrdiff_t>, 4> paddings;
  paddings[0] = std::make_pair(0, 0);
  paddings[1] = std::make_pair(2, 1);
  paddings[2] = std::make_pair(3, 4);
  paddings[3] = std::make_pair(0, 0);

  Eigen::DSizes<ptrdiff_t, 2> reshape_dims;
  reshape_dims[0] = 12;
  reshape_dims[1] = 84;

  Tensor<float, 2, DataLayout> result;
  result = tensor.pad(paddings).reshape(reshape_dims);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 6; ++j) {
      for (int k = 0; k < 12; ++k) {
        for (int l = 0; l < 7; ++l) {
          const float result_value = DataLayout == ColMajor ?
              result(i+2*j,k+12*l) : result(j+6*i,l+7*k);
          if (j >= 2 && j < 5 && k >= 3 && k < 8) {
            VERIFY_IS_EQUAL(result_value, tensor(i,j-2,k-3,l));
          } else {
            VERIFY_IS_EQUAL(result_value, 0.0f);
          }
        }
      }
    }
  }
}

void test_cxx11_tensor_padding()
{
  CALL_SUBTEST(test_simple_padding<ColMajor>());
  CALL_SUBTEST(test_simple_padding<RowMajor>());
  CALL_SUBTEST(test_padded_expr<ColMajor>());
  CALL_SUBTEST(test_padded_expr<RowMajor>());
}
