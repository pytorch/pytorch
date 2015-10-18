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

static void test_simple_lvalue_ref()
{
  Tensor<int, 1> input(6);
  input.setRandom();

  TensorRef<Tensor<int, 1>> ref3(input);
  TensorRef<Tensor<int, 1>> ref4 = input;

  VERIFY_IS_EQUAL(ref3.data(), input.data());
  VERIFY_IS_EQUAL(ref4.data(), input.data());

  for (int i = 0; i < 6; ++i) {
    VERIFY_IS_EQUAL(ref3(i), input(i));
    VERIFY_IS_EQUAL(ref4(i), input(i));
  }

  for (int i = 0; i < 6; ++i) {
    ref3.coeffRef(i) = i;
  }
  for (int i = 0; i < 6; ++i) {
    VERIFY_IS_EQUAL(input(i), i);
  }
  for (int i = 0; i < 6; ++i) {
    ref4.coeffRef(i) = -i * 2;
  }
  for (int i = 0; i < 6; ++i) {
    VERIFY_IS_EQUAL(input(i), -i*2);
  }
}


static void test_simple_rvalue_ref()
{
  Tensor<int, 1> input1(6);
  input1.setRandom();
  Tensor<int, 1> input2(6);
  input2.setRandom();

  TensorRef<Tensor<int, 1>> ref3(input1 + input2);
  TensorRef<Tensor<int, 1>> ref4 = input1 + input2;

  VERIFY_IS_NOT_EQUAL(ref3.data(), input1.data());
  VERIFY_IS_NOT_EQUAL(ref4.data(), input1.data());
  VERIFY_IS_NOT_EQUAL(ref3.data(), input2.data());
  VERIFY_IS_NOT_EQUAL(ref4.data(), input2.data());

  for (int i = 0; i < 6; ++i) {
    VERIFY_IS_EQUAL(ref3(i), input1(i) + input2(i));
    VERIFY_IS_EQUAL(ref4(i), input1(i) + input2(i));
  }
}


static void test_multiple_dims()
{
  Tensor<float, 3> input(3,5,7);
  input.setRandom();

  TensorRef<Tensor<float, 3>> ref(input);
  VERIFY_IS_EQUAL(ref.data(), input.data());
  VERIFY_IS_EQUAL(ref.dimension(0), 3);
  VERIFY_IS_EQUAL(ref.dimension(1), 5);
  VERIFY_IS_EQUAL(ref.dimension(2), 7);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(ref(i,j,k), input(i,j,k));
      }
    }
  }
}


static void test_slice()
{
  Tensor<float, 5> tensor(2,3,5,7,11);
  tensor.setRandom();

  Eigen::DSizes<ptrdiff_t, 5> indices(1,2,3,4,5);
  Eigen::DSizes<ptrdiff_t, 5> sizes(1,1,1,1,1);
  TensorRef<Tensor<float, 5>> slice = tensor.slice(indices, sizes);
  VERIFY_IS_EQUAL(slice(0,0,0,0,0), tensor(1,2,3,4,5));

  Eigen::DSizes<ptrdiff_t, 5> indices2(1,1,3,4,5);
  Eigen::DSizes<ptrdiff_t, 5> sizes2(1,1,2,2,3);
  slice = tensor.slice(indices2, sizes2);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        VERIFY_IS_EQUAL(slice(0,0,i,j,k), tensor(1,1,3+i,4+j,5+k));
      }
    }
  }

  Eigen::DSizes<ptrdiff_t, 5> indices3(0,0,0,0,0);
  Eigen::DSizes<ptrdiff_t, 5> sizes3(2,3,1,1,1);
  slice = tensor.slice(indices3, sizes3);
  VERIFY_IS_EQUAL(slice.data(), tensor.data());
}


static void test_ref_of_ref()
{
  Tensor<float, 3> input(3,5,7);
  input.setRandom();

  TensorRef<Tensor<float, 3>> ref(input);
  TensorRef<Tensor<float, 3>> ref_of_ref(ref);
  TensorRef<Tensor<float, 3>> ref_of_ref2;
  ref_of_ref2 = ref;

  VERIFY_IS_EQUAL(ref_of_ref.data(), input.data());
  VERIFY_IS_EQUAL(ref_of_ref.dimension(0), 3);
  VERIFY_IS_EQUAL(ref_of_ref.dimension(1), 5);
  VERIFY_IS_EQUAL(ref_of_ref.dimension(2), 7);

  VERIFY_IS_EQUAL(ref_of_ref2.data(), input.data());
  VERIFY_IS_EQUAL(ref_of_ref2.dimension(0), 3);
  VERIFY_IS_EQUAL(ref_of_ref2.dimension(1), 5);
  VERIFY_IS_EQUAL(ref_of_ref2.dimension(2), 7);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(ref_of_ref(i,j,k), input(i,j,k));
        VERIFY_IS_EQUAL(ref_of_ref2(i,j,k), input(i,j,k));
     }
    }
  }
}


static void test_ref_in_expr()
{
  Tensor<float, 3> input(3,5,7);
  input.setRandom();
  TensorRef<Tensor<float, 3>> input_ref(input);

  Tensor<float, 3> result(3,5,7);
  result.setRandom();
  TensorRef<Tensor<float, 3>> result_ref(result);

  Tensor<float, 3> bias(3,5,7);
  bias.setRandom();

  result_ref = input_ref + bias;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(result_ref(i,j,k), input(i,j,k) + bias(i,j,k));
        VERIFY_IS_NOT_EQUAL(result(i,j,k), input(i,j,k) + bias(i,j,k));
      }
    }
  }

  result = result_ref;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(result(i,j,k), input(i,j,k) + bias(i,j,k));
      }
    }
  }
}


static void test_coeff_ref()
{
  Tensor<float, 5> tensor(2,3,5,7,11);
  tensor.setRandom();
  Tensor<float, 5> original = tensor;

  TensorRef<Tensor<float, 4>> slice = tensor.chip(7, 4);
  slice.coeffRef(0, 0, 0, 0) = 1.0f;
  slice.coeffRef(1, 0, 0, 0) += 2.0f;

  VERIFY_IS_EQUAL(tensor(0,0,0,0,7), 1.0f);
  VERIFY_IS_EQUAL(tensor(1,0,0,0,7), original(1,0,0,0,7) + 2.0f);
}


static void test_nested_ops_with_ref()
{
  Tensor<float, 4> t(2, 3, 5, 7);
  t.setRandom();
  TensorMap<Tensor<const float, 4> > m(t.data(), 2, 3, 5, 7);
  array<std::pair<ptrdiff_t, ptrdiff_t>, 4> paddings;
  paddings[0] = std::make_pair(0, 0);
  paddings[1] = std::make_pair(2, 1);
  paddings[2] = std::make_pair(3, 4);
  paddings[3] = std::make_pair(0, 0);
  DSizes<Eigen::DenseIndex, 4> shuffle_dims(0, 1, 2, 3);
  TensorRef<Tensor<const float, 4> > ref(m.pad(paddings));
  array<std::pair<ptrdiff_t, ptrdiff_t>, 4> trivial;
  trivial[0] = std::make_pair(0, 0);
  trivial[1] = std::make_pair(0, 0);
  trivial[2] = std::make_pair(0, 0);
  trivial[3] = std::make_pair(0, 0);
  Tensor<float, 4> padded = ref.shuffle(shuffle_dims).pad(trivial);
  VERIFY_IS_EQUAL(padded.dimension(0), 2+0);
  VERIFY_IS_EQUAL(padded.dimension(1), 3+3);
  VERIFY_IS_EQUAL(padded.dimension(2), 5+7);
  VERIFY_IS_EQUAL(padded.dimension(3), 7+0);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 6; ++j) {
      for (int k = 0; k < 12; ++k) {
        for (int l = 0; l < 7; ++l) {
          if (j >= 2 && j < 5 && k >= 3 && k < 8) {
            VERIFY_IS_EQUAL(padded(i,j,k,l), t(i,j-2,k-3,l));
          } else {
            VERIFY_IS_EQUAL(padded(i,j,k,l), 0.0f);
          }
        }
      }
    }
  }
}


void test_cxx11_tensor_ref()
{
  CALL_SUBTEST(test_simple_lvalue_ref());
  CALL_SUBTEST(test_simple_rvalue_ref());
  CALL_SUBTEST(test_multiple_dims());
  CALL_SUBTEST(test_slice());
  CALL_SUBTEST(test_ref_of_ref());
  CALL_SUBTEST(test_ref_in_expr());
  CALL_SUBTEST(test_coeff_ref());
  CALL_SUBTEST(test_nested_ops_with_ref());
}
