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
#include <qnnpack/requantization.h>

class ClampMicrokernelTester {
 public:
  inline ClampMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline ClampMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline ClampMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline ClampMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline ClampMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(pytorch_u8clamp_ukernel_function u8clamp) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> x(n());
    std::vector<uint8_t> y(n());
    std::vector<uint8_t> yRef(n());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(u8rng));
      } else {
        std::fill(y.begin(), y.end(), 0xA5);
      }
      const uint8_t* xData = inplace() ? y.data() : x.data();

      /* Prepare clamping parameters */
      const union pytorch_qnnp_u8_clamping_params clampingParams =
          pytorch_qnnp_compute_u8_clamping_params(qmin(), qmax());

      /* Compute reference results */
      for (size_t i = 0; i < n(); i++) {
        yRef[i] = std::max(std::min(xData[i], qmax()), qmin());
      }

      /* Call optimized micro-kernel */
      u8clamp(n(), xData, y.data(), &clampingParams);

      /* Verify results */
      for (size_t i = 0; i < n(); i++) {
        ASSERT_LE(uint32_t(y[i]), uint32_t(qmax()))
            << "at position " << i << ", n = " << n();
        ASSERT_GE(uint32_t(y[i]), uint32_t(qmin()))
            << "at position " << i << ", n = " << n();
        ASSERT_EQ(uint32_t(yRef[i]), uint32_t(y[i]))
            << "at position " << i << ", n = " << n() << ", qmin = " << qmin()
            << ", qmax = " << qmax();
      }
    }
  }

 private:
  size_t n_{1};
  bool inplace_{false};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
