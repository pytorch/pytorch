/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdlib>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <qnnpack/params.h>

class ZipMicrokernelTester {
 public:
  inline ZipMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline ZipMicrokernelTester& g(size_t g) {
    assert(g != 0);
    this->g_ = g;
    return *this;
  }

  inline size_t g() const {
    return this->g_;
  }

  inline ZipMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(pytorch_xzipc_ukernel_function xzip) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> x(n() * g());
    std::vector<uint8_t> y(g() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      /* Call optimized micro-kernel */
      xzip(n(), x.data(), y.data());

      /* Verify results */
      for (size_t i = 0; i < n(); i++) {
        for (size_t j = 0; j < g(); j++) {
          ASSERT_EQ(uint32_t(y[i * g() + j]), uint32_t(x[j * n() + i]))
              << "at element " << i << ", group " << j;
        }
      }
    }
  }

  void test(pytorch_xzipv_ukernel_function xzip) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> x(n() * g());
    std::vector<uint8_t> y(g() * n());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      /* Call optimized micro-kernel */
      xzip(n(), g(), x.data(), y.data());

      /* Verify results */
      for (size_t i = 0; i < n(); i++) {
        for (size_t j = 0; j < g(); j++) {
          ASSERT_EQ(uint32_t(y[i * g() + j]), uint32_t(x[j * n() + i]))
              << "at element " << i << ", group " << j;
        }
      }
    }
  }

 private:
  size_t n_{1};
  size_t g_{1};
  size_t iterations_{3};
};
