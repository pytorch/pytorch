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

static void test_default()
{
  Tensor<float, 1> vec(6);
  vec.setRandom();

  // Fixme: we should check that the generated numbers follow a uniform
  // distribution instead.
  for (int i = 1; i < 6; ++i) {
    VERIFY_IS_NOT_EQUAL(vec(i), vec(i-1));
  }
}

static void test_normal()
{
  Tensor<float, 1> vec(6);
  vec.setRandom<Eigen::internal::NormalRandomGenerator<float>>();

  // Fixme: we should check that the generated numbers follow a gaussian
  // distribution instead.
  for (int i = 1; i < 6; ++i) {
    VERIFY_IS_NOT_EQUAL(vec(i), vec(i-1));
  }
}


struct MyGenerator {
  MyGenerator() { }
  MyGenerator(const MyGenerator&) { }

  // Return a random value to be used.  "element_location" is the
  // location of the entry to set in the tensor, it can typically
  // be ignored.
  int operator()(Eigen::DenseIndex element_location, Eigen::DenseIndex /*unused*/ = 0) const {
    return static_cast<int>(3 * element_location);
  }

  // Same as above but generates several numbers at a time.
  typename internal::packet_traits<int>::type packetOp(
      Eigen::DenseIndex packet_location, Eigen::DenseIndex /*unused*/ = 0) const {
    const int packetSize = internal::packet_traits<int>::size;
    EIGEN_ALIGN_MAX int values[packetSize];
    for (int i = 0; i < packetSize; ++i) {
      values[i] = static_cast<int>(3 * (packet_location + i));
    }
    return internal::pload<typename internal::packet_traits<int>::type>(values);
  }
};


static void test_custom()
{
  Tensor<int, 1> vec(6);
  vec.setRandom<MyGenerator>();

  for (int i = 0; i < 6; ++i) {
    VERIFY_IS_EQUAL(vec(i), 3*i);
  }
}

void test_cxx11_tensor_random()
{
  CALL_SUBTEST(test_default());
  CALL_SUBTEST(test_normal());
  CALL_SUBTEST(test_custom());
}
