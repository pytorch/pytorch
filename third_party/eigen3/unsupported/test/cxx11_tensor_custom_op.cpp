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


struct InsertZeros {
  DSizes<DenseIndex, 2> dimensions(const Tensor<float, 2>& input) const {
    DSizes<DenseIndex, 2> result;
    result[0] = input.dimension(0) * 2;
    result[1] = input.dimension(1) * 2;
    return result;
  }

  template <typename Output, typename Device>
  void eval(const Tensor<float, 2>& input, Output& output, const Device& device) const
  {
    array<DenseIndex, 2> strides;
    strides[0] = 2;
    strides[1] = 2;
    output.stride(strides).device(device) = input;

    Eigen::DSizes<DenseIndex, 2> offsets(1,1);
    Eigen::DSizes<DenseIndex, 2> extents(output.dimension(0)-1, output.dimension(1)-1);
    output.slice(offsets, extents).stride(strides).device(device) = input.constant(0.0f);
  }
};

static void test_custom_unary_op()
{
  Tensor<float, 2> tensor(3,5);
  tensor.setRandom();

  Tensor<float, 2> result = tensor.customOp(InsertZeros());
  VERIFY_IS_EQUAL(result.dimension(0), 6);
  VERIFY_IS_EQUAL(result.dimension(1), 10);

  for (int i = 0; i < 6; i+=2) {
    for (int j = 0; j < 10; j+=2) {
      VERIFY_IS_EQUAL(result(i, j), tensor(i/2, j/2));
    }
  }
  for (int i = 1; i < 6; i+=2) {
    for (int j = 1; j < 10; j+=2) {
      VERIFY_IS_EQUAL(result(i, j), 0);
    }
  }
}


struct BatchMatMul {
  DSizes<DenseIndex, 3> dimensions(const Tensor<float, 3>& input1, const Tensor<float, 3>& input2) const {
    DSizes<DenseIndex, 3> result;
    result[0] = input1.dimension(0);
    result[1] = input2.dimension(1);
    result[2] = input2.dimension(2);
    return result;
  }

  template <typename Output, typename Device>
  void eval(const Tensor<float, 3>& input1, const Tensor<float, 3>& input2,
            Output& output, const Device& device) const
  {
    typedef Tensor<float, 3>::DimensionPair DimPair;
    array<DimPair, 1> dims;
    dims[0] = DimPair(1, 0);
    for (int i = 0; i < output.dimension(2); ++i) {
      output.template chip<2>(i).device(device) = input1.chip<2>(i).contract(input2.chip<2>(i), dims);
    }
  }
};


static void test_custom_binary_op()
{
  Tensor<float, 3> tensor1(2,3,5);
  tensor1.setRandom();
  Tensor<float, 3> tensor2(3,7,5);
  tensor2.setRandom();

  Tensor<float, 3> result = tensor1.customOp(tensor2, BatchMatMul());
  for (int i = 0; i < 5; ++i) {
    typedef Tensor<float, 3>::DimensionPair DimPair;
    array<DimPair, 1> dims;
    dims[0] = DimPair(1, 0);
    Tensor<float, 2> reference = tensor1.chip<2>(i).contract(tensor2.chip<2>(i), dims);
    TensorRef<Tensor<float, 2> > val = result.chip<2>(i);
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_APPROX(val(j, k), reference(j, k));
      }
    }
  }
}


void test_cxx11_tensor_custom_op()
{
  CALL_SUBTEST(test_custom_unary_op());
  CALL_SUBTEST(test_custom_binary_op());
}
