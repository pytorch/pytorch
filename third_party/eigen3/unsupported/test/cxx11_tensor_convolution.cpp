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
using Eigen::DefaultDevice;

template <int DataLayout>
static void test_evals()
{
  Tensor<float, 2, DataLayout> input(3, 3);
  Tensor<float, 1, DataLayout> kernel(2);

  input.setRandom();
  kernel.setRandom();

  Tensor<float, 2, DataLayout> result(2,3);
  result.setZero();
  Eigen::array<Tensor<float, 2>::Index, 1> dims3{{0}};

  typedef TensorEvaluator<decltype(input.convolve(kernel, dims3)), DefaultDevice> Evaluator;
  Evaluator eval(input.convolve(kernel, dims3), DefaultDevice());
  eval.evalTo(result.data());
  EIGEN_STATIC_ASSERT(Evaluator::NumDims==2ul, YOU_MADE_A_PROGRAMMING_MISTAKE);
  VERIFY_IS_EQUAL(eval.dimensions()[0], 2);
  VERIFY_IS_EQUAL(eval.dimensions()[1], 3);

  VERIFY_IS_APPROX(result(0,0), input(0,0)*kernel(0) + input(1,0)*kernel(1));  // index 0
  VERIFY_IS_APPROX(result(0,1), input(0,1)*kernel(0) + input(1,1)*kernel(1));  // index 2
  VERIFY_IS_APPROX(result(0,2), input(0,2)*kernel(0) + input(1,2)*kernel(1));  // index 4
  VERIFY_IS_APPROX(result(1,0), input(1,0)*kernel(0) + input(2,0)*kernel(1));  // index 1
  VERIFY_IS_APPROX(result(1,1), input(1,1)*kernel(0) + input(2,1)*kernel(1));  // index 3
  VERIFY_IS_APPROX(result(1,2), input(1,2)*kernel(0) + input(2,2)*kernel(1));  // index 5
}

template <int DataLayout>
static void test_expr()
{
  Tensor<float, 2, DataLayout> input(3, 3);
  Tensor<float, 2, DataLayout> kernel(2, 2);
  input.setRandom();
  kernel.setRandom();

  Tensor<float, 2, DataLayout> result(2,2);
  Eigen::array<ptrdiff_t, 2> dims;
  dims[0] = 0;
  dims[1] = 1;
  result = input.convolve(kernel, dims);

  VERIFY_IS_APPROX(result(0,0), input(0,0)*kernel(0,0) + input(0,1)*kernel(0,1) +
                                input(1,0)*kernel(1,0) + input(1,1)*kernel(1,1));
  VERIFY_IS_APPROX(result(0,1), input(0,1)*kernel(0,0) + input(0,2)*kernel(0,1) +
                                input(1,1)*kernel(1,0) + input(1,2)*kernel(1,1));
  VERIFY_IS_APPROX(result(1,0), input(1,0)*kernel(0,0) + input(1,1)*kernel(0,1) +
                                input(2,0)*kernel(1,0) + input(2,1)*kernel(1,1));
  VERIFY_IS_APPROX(result(1,1), input(1,1)*kernel(0,0) + input(1,2)*kernel(0,1) +
                                input(2,1)*kernel(1,0) + input(2,2)*kernel(1,1));
}

template <int DataLayout>
static void test_modes() {
  Tensor<float, 1, DataLayout> input(3);
  Tensor<float, 1, DataLayout> kernel(3);
  input(0) = 1.0f;
  input(1) = 2.0f;
  input(2) = 3.0f;
  kernel(0) = 0.5f;
  kernel(1) = 1.0f;
  kernel(2) = 0.0f;

  Eigen::array<ptrdiff_t, 1> dims;
  dims[0] = 0;
  Eigen::array<std::pair<ptrdiff_t, ptrdiff_t>, 1> padding;

  // Emulate VALID mode (as defined in
  // http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html).
  padding[0] = std::make_pair(0, 0);
  Tensor<float, 1, DataLayout> valid(1);
  valid = input.pad(padding).convolve(kernel, dims);
  VERIFY_IS_EQUAL(valid.dimension(0), 1);
  VERIFY_IS_APPROX(valid(0), 2.5f);

  // Emulate SAME mode (as defined in
  // http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html).
  padding[0] = std::make_pair(1, 1);
  Tensor<float, 1, DataLayout> same(3);
  same = input.pad(padding).convolve(kernel, dims);
  VERIFY_IS_EQUAL(same.dimension(0), 3);
  VERIFY_IS_APPROX(same(0), 1.0f);
  VERIFY_IS_APPROX(same(1), 2.5f);
  VERIFY_IS_APPROX(same(2), 4.0f);

  // Emulate FULL mode (as defined in
  // http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html).
  padding[0] = std::make_pair(2, 2);
  Tensor<float, 1, DataLayout> full(5);
  full = input.pad(padding).convolve(kernel, dims);
  VERIFY_IS_EQUAL(full.dimension(0), 5);
  VERIFY_IS_APPROX(full(0), 0.0f);
  VERIFY_IS_APPROX(full(1), 1.0f);
  VERIFY_IS_APPROX(full(2), 2.5f);
  VERIFY_IS_APPROX(full(3), 4.0f);
  VERIFY_IS_APPROX(full(4), 1.5f);
}

template <int DataLayout>
static void test_strides() {
  Tensor<float, 1, DataLayout> input(13);
  Tensor<float, 1, DataLayout> kernel(3);
  input.setRandom();
  kernel.setRandom();

  Eigen::array<ptrdiff_t, 1> dims;
  dims[0] = 0;
  Eigen::array<ptrdiff_t, 1> stride_of_3;
  stride_of_3[0] = 3;
  Eigen::array<ptrdiff_t, 1> stride_of_2;
  stride_of_2[0] = 2;

  Tensor<float, 1, DataLayout> result;
  result = input.stride(stride_of_3).convolve(kernel, dims).stride(stride_of_2);

  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_APPROX(result(0), (input(0)*kernel(0) + input(3)*kernel(1) +
                               input(6)*kernel(2)));
  VERIFY_IS_APPROX(result(1), (input(6)*kernel(0) + input(9)*kernel(1) +
                               input(12)*kernel(2)));
}

void test_cxx11_tensor_convolution()
{
  CALL_SUBTEST(test_evals<ColMajor>());
  CALL_SUBTEST(test_evals<RowMajor>());
  CALL_SUBTEST(test_expr<ColMajor>());
  CALL_SUBTEST(test_expr<RowMajor>());
  CALL_SUBTEST(test_modes<ColMajor>());
  CALL_SUBTEST(test_modes<RowMajor>());
  CALL_SUBTEST(test_strides<ColMajor>());
  CALL_SUBTEST(test_strides<RowMajor>());
}
