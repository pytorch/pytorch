/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <qnnpack/params.h>

class RMaxMicrokernelTester {
 public:
  inline RMaxMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline RMaxMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(pytorch_u8rmax_ukernel_function u8rmax) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> x(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));

      /* Compute reference results */
      uint8_t yRef = 0;
      for (size_t i = 0; i < n(); i++) {
        yRef = std::max(yRef, x[i]);
      }

      /* Call optimized micro-kernel */
      const uint8_t y = u8rmax(n(), x.data());

      /* Verify results */
      ASSERT_EQ(yRef, y) << "n = " << n();
    }
  }

 private:
  size_t n_{1};
  size_t iterations_{15};
};
