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
using Eigen::array;

template <int DataLayout>
static void test_simple_shuffling()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();
  array<ptrdiff_t, 4> shuffles;
  shuffles[0] = 0;
  shuffles[1] = 1;
  shuffles[2] = 2;
  shuffles[3] = 3;

  Tensor<float, 4, DataLayout> no_shuffle;
  no_shuffle = tensor.shuffle(shuffles);

  VERIFY_IS_EQUAL(no_shuffle.dimension(0), 2);
  VERIFY_IS_EQUAL(no_shuffle.dimension(1), 3);
  VERIFY_IS_EQUAL(no_shuffle.dimension(2), 5);
  VERIFY_IS_EQUAL(no_shuffle.dimension(3), 7);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i,j,k,l), no_shuffle(i,j,k,l));
        }
      }
    }
  }

  shuffles[0] = 2;
  shuffles[1] = 3;
  shuffles[2] = 1;
  shuffles[3] = 0;
  Tensor<float, 4, DataLayout> shuffle;
  shuffle = tensor.shuffle(shuffles);

  VERIFY_IS_EQUAL(shuffle.dimension(0), 5);
  VERIFY_IS_EQUAL(shuffle.dimension(1), 7);
  VERIFY_IS_EQUAL(shuffle.dimension(2), 3);
  VERIFY_IS_EQUAL(shuffle.dimension(3), 2);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i,j,k,l), shuffle(k,l,j,i));
        }
      }
    }
  }
}


template <int DataLayout>
static void test_expr_shuffling()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();

  array<ptrdiff_t, 4> shuffles;
  shuffles[0] = 2;
  shuffles[1] = 3;
  shuffles[2] = 1;
  shuffles[3] = 0;
  Tensor<float, 4, DataLayout> expected;
  expected = tensor.shuffle(shuffles);

  Tensor<float, 4, DataLayout> result(5,7,3,2);

  array<int, 4> src_slice_dim{{2,3,1,7}};
  array<int, 4> src_slice_start{{0,0,0,0}};
  array<int, 4> dst_slice_dim{{1,7,3,2}};
  array<int, 4> dst_slice_start{{0,0,0,0}};

  for (int i = 0; i < 5; ++i) {
    result.slice(dst_slice_start, dst_slice_dim) =
        tensor.slice(src_slice_start, src_slice_dim).shuffle(shuffles);
    src_slice_start[2] += 1;
    dst_slice_start[0] += 1;
  }

  VERIFY_IS_EQUAL(result.dimension(0), 5);
  VERIFY_IS_EQUAL(result.dimension(1), 7);
  VERIFY_IS_EQUAL(result.dimension(2), 3);
  VERIFY_IS_EQUAL(result.dimension(3), 2);

  for (int i = 0; i < expected.dimension(0); ++i) {
    for (int j = 0; j < expected.dimension(1); ++j) {
      for (int k = 0; k < expected.dimension(2); ++k) {
        for (int l = 0; l < expected.dimension(3); ++l) {
          VERIFY_IS_EQUAL(result(i,j,k,l), expected(i,j,k,l));
        }
      }
    }
  }

  dst_slice_start[0] = 0;
  result.setRandom();
  for (int i = 0; i < 5; ++i) {
    result.slice(dst_slice_start, dst_slice_dim) =
        tensor.shuffle(shuffles).slice(dst_slice_start, dst_slice_dim);
    dst_slice_start[0] += 1;
  }

  for (int i = 0; i < expected.dimension(0); ++i) {
    for (int j = 0; j < expected.dimension(1); ++j) {
      for (int k = 0; k < expected.dimension(2); ++k) {
        for (int l = 0; l < expected.dimension(3); ++l) {
          VERIFY_IS_EQUAL(result(i,j,k,l), expected(i,j,k,l));
        }
      }
    }
  }
}


template <int DataLayout>
static void test_shuffling_as_value()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();
  array<ptrdiff_t, 4> shuffles;
  shuffles[2] = 0;
  shuffles[3] = 1;
  shuffles[1] = 2;
  shuffles[0] = 3;
  Tensor<float, 4, DataLayout> shuffle(5,7,3,2);
  shuffle.shuffle(shuffles) = tensor;

  VERIFY_IS_EQUAL(shuffle.dimension(0), 5);
  VERIFY_IS_EQUAL(shuffle.dimension(1), 7);
  VERIFY_IS_EQUAL(shuffle.dimension(2), 3);
  VERIFY_IS_EQUAL(shuffle.dimension(3), 2);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i,j,k,l), shuffle(k,l,j,i));
        }
      }
    }
  }

  array<ptrdiff_t, 4> no_shuffle;
  no_shuffle[0] = 0;
  no_shuffle[1] = 1;
  no_shuffle[2] = 2;
  no_shuffle[3] = 3;
  Tensor<float, 4, DataLayout> shuffle2(5,7,3,2);
  shuffle2.shuffle(shuffles) = tensor.shuffle(no_shuffle);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 2; ++l) {
          VERIFY_IS_EQUAL(shuffle2(i,j,k,l), shuffle(i,j,k,l));
        }
      }
    }
  }
}


template <int DataLayout>
static void test_shuffle_unshuffle()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();

  // Choose a random permutation.
  array<ptrdiff_t, 4> shuffles;
  for (int i = 0; i < 4; ++i) {
    shuffles[i] = i;
  }
  array<ptrdiff_t, 4> shuffles_inverse;
  for (int i = 0; i < 4; ++i) {
    const ptrdiff_t index = internal::random<ptrdiff_t>(i, 3);
    shuffles_inverse[shuffles[index]] = i;
    std::swap(shuffles[i], shuffles[index]);
  }

  Tensor<float, 4, DataLayout> shuffle;
  shuffle = tensor.shuffle(shuffles).shuffle(shuffles_inverse);

  VERIFY_IS_EQUAL(shuffle.dimension(0), 2);
  VERIFY_IS_EQUAL(shuffle.dimension(1), 3);
  VERIFY_IS_EQUAL(shuffle.dimension(2), 5);
  VERIFY_IS_EQUAL(shuffle.dimension(3), 7);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i,j,k,l), shuffle(i,j,k,l));
        }
      }
    }
  }
}


void test_cxx11_tensor_shuffling()
{
  CALL_SUBTEST(test_simple_shuffling<ColMajor>());
  CALL_SUBTEST(test_simple_shuffling<RowMajor>());
  CALL_SUBTEST(test_expr_shuffling<ColMajor>());
  CALL_SUBTEST(test_expr_shuffling<RowMajor>());
  CALL_SUBTEST(test_shuffling_as_value<ColMajor>());
  CALL_SUBTEST(test_shuffling_as_value<RowMajor>());
  CALL_SUBTEST(test_shuffle_unshuffle<ColMajor>());
  CALL_SUBTEST(test_shuffle_unshuffle<RowMajor>());
}
