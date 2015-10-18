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


static void test_dynamic_size()
{
  Eigen::DSizes<int, 3> dimensions(2,3,7);

  VERIFY_IS_EQUAL((int)Eigen::internal::array_get<0>(dimensions), 2);
  VERIFY_IS_EQUAL((int)Eigen::internal::array_get<1>(dimensions), 3);
  VERIFY_IS_EQUAL((int)Eigen::internal::array_get<2>(dimensions), 7);
  VERIFY_IS_EQUAL(dimensions.TotalSize(), 2*3*7);
  VERIFY_IS_EQUAL((int)dimensions[0], 2);
  VERIFY_IS_EQUAL((int)dimensions[1], 3);
  VERIFY_IS_EQUAL((int)dimensions[2], 7);
}

static void test_fixed_size()
{
  Eigen::Sizes<2,3,7> dimensions;

  VERIFY_IS_EQUAL((int)Eigen::internal::array_get<0>(dimensions), 2);
  VERIFY_IS_EQUAL((int)Eigen::internal::array_get<1>(dimensions), 3);
  VERIFY_IS_EQUAL((int)Eigen::internal::array_get<2>(dimensions), 7);
  VERIFY_IS_EQUAL(dimensions.TotalSize(), 2*3*7);
}


static void test_match()
{
  Eigen::DSizes<int, 3> dyn(2,3,7);
  Eigen::Sizes<2,3,7> stat;
  VERIFY_IS_EQUAL(Eigen::dimensions_match(dyn, stat), true);

  Eigen::DSizes<int, 3> dyn1(2,3,7);
  Eigen::DSizes<int, 2> dyn2(2,3);
  VERIFY_IS_EQUAL(Eigen::dimensions_match(dyn1, dyn2), false);
}


void test_cxx11_tensor_dimension()
{
  CALL_SUBTEST(test_dynamic_size());
  CALL_SUBTEST(test_fixed_size());
  CALL_SUBTEST(test_match());
}
