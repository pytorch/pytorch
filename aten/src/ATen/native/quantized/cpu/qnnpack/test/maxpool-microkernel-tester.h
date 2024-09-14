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

class MaxPoolMicrokernelTester {
 public:
  inline MaxPoolMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline MaxPoolMicrokernelTester& s(size_t s) {
    assert(s != 0);
    this->s_ = s;
    return *this;
  }

  inline size_t s() const {
    return this->s_;
  }

  inline MaxPoolMicrokernelTester& kh(size_t kh) {
    assert(kh != 0);
    this->kh_ = kh;
    return *this;
  }

  inline size_t kh() const {
    return this->kh_;
  }

  inline MaxPoolMicrokernelTester& kw(size_t kw) {
    assert(kw != 0);
    this->kw_ = kw;
    return *this;
  }

  inline size_t kw() const {
    return this->kw_;
  }

  inline size_t ks() const {
    return kh() * kw();
  }

  inline size_t packedKs() const {
    if (kc() < kr()) {
      return ks();
    } else if (ks() <= mr()) {
      return mr();
    } else {
      return (ks() - mr()) % qr() == 0
          ? ks()
          : ((ks() - mr()) / qr() + 1) * qr() + mr();
    }
  }

  inline MaxPoolMicrokernelTester& mr(size_t mr) {
    assert(mr != 0);
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline MaxPoolMicrokernelTester& qr(size_t qr) {
    assert(qr != 0);
    this->qr_ = qr;
    return *this;
  }

  inline size_t qr() const {
    return this->qr_;
  }

  inline MaxPoolMicrokernelTester& kc(size_t kc) {
    assert(kc != 0);
    this->kc_ = kc;
    return *this;
  }

  inline size_t kc() const {
    return this->kc_;
  }

  inline MaxPoolMicrokernelTester& kr(size_t kr) {
    assert(kr != 0);
    this->kr_ = kr;
    return *this;
  }

  inline size_t kr() const {
    return this->kr_;
  }

  inline size_t packedN() const {
    return kc() % kr() == 0 ? kc() : (kc() / kr() + 1) * kr();
  }

  inline MaxPoolMicrokernelTester& xStride(size_t xStride) {
    assert(xStride != 0);
    this->xStride_ = xStride;
    return *this;
  }

  inline size_t xStride() const {
    if (this->xStride_ == 0) {
      return kc();
    } else {
      assert(this->xStride_ >= kc());
      return this->xStride_;
    }
  }

  inline MaxPoolMicrokernelTester& yStride(size_t yStride) {
    assert(yStride != 0);
    this->yStride_ = yStride;
    return *this;
  }

  inline size_t yStride() const {
    if (this->yStride_ == 0) {
      return kc();
    } else {
      assert(this->yStride_ >= kc());
      return this->yStride_;
    }
  }

  inline MaxPoolMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline MaxPoolMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline MaxPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(pytorch_u8maxpool_ukernel_function u8maxpool) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<const uint8_t*> indirectX(packedKs() + (n() * s() - 1) * kh());
    std::vector<uint8_t> x((indirectX.size() - 1) * xStride() + kc());

    std::vector<uint8_t> zero(kc());
    std::vector<uint8_t> y((n() - 1) * yStride() + kc());
    std::vector<uint8_t> yRef(n() * kc());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      for (size_t i = 0; i < indirectX.size(); i++) {
        indirectX[i] = x.data() + i * xStride();
      }
      std::shuffle(indirectX.begin(), indirectX.end(), rng);

      /* Prepare quantization parameters */
      const union pytorch_qnnp_u8_clamping_params clampingParams =
          pytorch_qnnp_compute_u8_clamping_params(qmin(), qmax());

      /* Compute reference results */
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          uint8_t maxValue = 0;
          for (size_t j = 0; j < ks(); j++) {
            maxValue = std::max(maxValue, indirectX[i * s() * kh() + j][k]);
          }
          maxValue = std::min(maxValue, qmax());
          maxValue = std::max(maxValue, qmin());
          yRef[i * kc() + k] = maxValue;
        }
      }

      /* Call optimized micro-kernel */
      u8maxpool(
          n(),
          ks(),
          kc(),
          indirectX.data(),
          y.data(),
          (kh() * s() - packedKs()) * sizeof(void*),
          (yStride() - kc()) * sizeof(uint8_t),
          &clampingParams);

      /* Verify results */
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          ASSERT_EQ(
              uint32_t(yRef[i * kc() + k]), uint32_t(y[i * yStride() + k]))
              << "at pixel " << i << ", channel " << k << ", n = " << n()
              << ", ks = " << kh() << "x" << kw() << " (" << ks()
              << "), kc = " << kc();
        }
      }
    }
  }

 private:
  size_t n_{1};
  size_t s_{1};
  size_t kh_{1};
  size_t kw_{1};
  size_t mr_{1};
  size_t qr_{1};
  size_t kc_{1};
  size_t kr_{1};
  size_t xStride_{0};
  size_t yStride_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
