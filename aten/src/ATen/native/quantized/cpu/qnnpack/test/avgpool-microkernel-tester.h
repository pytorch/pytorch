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
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <qnnpack/AlignedAllocator.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

class AvgPoolMicrokernelTester {
 public:
  inline AvgPoolMicrokernelTester& n(size_t n) {
    assert(n != 0);
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline AvgPoolMicrokernelTester& s(size_t s) {
    assert(s != 0);
    this->s_ = s;
    return *this;
  }

  inline size_t s() const {
    return this->s_;
  }

  inline AvgPoolMicrokernelTester& kh(size_t kh) {
    assert(kh != 0);
    this->kh_ = kh;
    return *this;
  }

  inline size_t kh() const {
    return this->kh_;
  }

  inline AvgPoolMicrokernelTester& kw(size_t kw) {
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

  inline AvgPoolMicrokernelTester& mr(size_t mr) {
    assert(mr != 0);
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline AvgPoolMicrokernelTester& qr(size_t qr) {
    assert(qr != 0);
    this->qr_ = qr;
    return *this;
  }

  inline size_t qr() const {
    return this->qr_;
  }

  inline AvgPoolMicrokernelTester& kc(size_t kc) {
    assert(kc != 0);
    this->kc_ = kc;
    return *this;
  }

  inline size_t kc() const {
    return this->kc_;
  }

  inline AvgPoolMicrokernelTester& kr(size_t kr) {
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

  inline AvgPoolMicrokernelTester& xStride(size_t xStride) {
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

  inline AvgPoolMicrokernelTester& yStride(size_t yStride) {
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

  inline AvgPoolMicrokernelTester& xScale(float xScale) {
    assert(xScale > 0.0f);
    assert(std::isnormal(xScale));
    this->xScale_ = xScale;
    return *this;
  }

  inline float xScale() const {
    return this->xScale_;
  }

  inline AvgPoolMicrokernelTester& xZeroPoint(uint8_t xZeroPoint) {
    this->xZeroPoint_ = xZeroPoint;
    return *this;
  }

  inline uint8_t xZeroPoint() const {
    return this->xZeroPoint_;
  }

  inline AvgPoolMicrokernelTester& yScale(float yScale) {
    assert(yScale > 0.0f);
    assert(std::isnormal(yScale));
    this->yScale_ = yScale;
    return *this;
  }

  inline float yScale() const {
    return this->yScale_;
  }

  inline AvgPoolMicrokernelTester& yZeroPoint(uint8_t yZeroPoint) {
    this->yZeroPoint_ = yZeroPoint;
    return *this;
  }

  inline uint8_t yZeroPoint() const {
    return this->yZeroPoint_;
  }

  inline AvgPoolMicrokernelTester& yMin(uint8_t yMin) {
    this->yMin_ = yMin;
    return *this;
  }

  inline uint8_t yMin() const {
    return this->yMin_;
  }

  inline AvgPoolMicrokernelTester& yMax(uint8_t yMax) {
    this->yMax_ = yMax;
    return *this;
  }

  inline uint8_t yMax() const {
    return this->yMax_;
  }

  inline AvgPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(pytorch_q8avgpool_up_ukernel_function q8avgpool) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<const uint8_t*> indirectX(packedKs() + (n() * s() - 1) * kh());
    std::vector<uint8_t> x((indirectX.size() - 1) * xStride() + kc());

    std::vector<uint8_t> zero(kc());
    std::vector<uint8_t> y((n() - 1) * yStride() + kc());
    std::vector<uint8_t> yRef(n() * kc());
    std::vector<float> yFP(n() * kc());
    std::vector<int32_t> yAcc(n() * kc());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      for (size_t i = 0; i < indirectX.size(); i++) {
        indirectX[i] = x.data() + i * xStride();
      }
      std::shuffle(indirectX.begin(), indirectX.end(), rng);

      /* Prepare quantization parameters */
      const union pytorch_qnnp_avgpool_quantization_params quantizationParams =
          pytorch_qnnp_compute_avgpool_quantization_params(
              -int32_t(xZeroPoint()) * int32_t(ks()),
              xScale() / (yScale() * float(ks())),
              yZeroPoint(),
              yMin(),
              yMax());
      const union pytorch_qnnp_avgpool_quantization_params
          scalarQuantizationParams =
              pytorch_qnnp_compute_scalar_avgpool_quantization_params(
                  -int32_t(xZeroPoint()) * int32_t(ks()),
                  xScale() / (yScale() * float(ks())),
                  yZeroPoint(),
                  yMin(),
                  yMax());

      /* Compute reference results */
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          int32_t acc = scalarQuantizationParams.scalar.bias;
          for (size_t j = 0; j < ks(); j++) {
            acc += indirectX[i * s() * kh() + j][k];
          }
          yAcc[i * kc() + k] = acc;
          yRef[i * kc() + k] =
              pytorch_qnnp_avgpool_quantize(acc, scalarQuantizationParams);
          yFP[i * kc() + k] =
              float(acc) * (xScale() / (yScale() * float(ks()))) +
              float(yZeroPoint());
          yFP[i * kc() + k] = std::min<float>(yFP[i * kc() + k], float(yMax()));
          yFP[i * kc() + k] = std::max<float>(yFP[i * kc() + k], float(yMin()));
        }
      }

      /* Call optimized micro-kernel */
      q8avgpool(
          n(),
          ks(),
          kc(),
          indirectX.data(),
          zero.data(),
          y.data(),
          kh() * s() * sizeof(void*),
          (yStride() - kc()) * sizeof(uint8_t),
          &quantizationParams);

      /* Verify results */
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          ASSERT_LE(uint32_t(y[i * yStride() + k]), uint32_t(yMax()))
              << "at pixel " << i << ", channel " << k << ", n = " << n()
              << ", kc = " << kc();
          ASSERT_GE(uint32_t(y[i * yStride() + k]), uint32_t(yMin()))
              << "at pixel " << i << ", channel " << k << ", n = " << n()
              << ", kc = " << kc();
          ASSERT_NEAR(
              float(int32_t(y[i * yStride() + k])), yFP[i * kc() + k], 0.5f)
              << "at pixel " << i << ", channel " << k << ", n = " << n()
              << ", ks = " << kh() << "x" << kw() << " (" << ks()
              << "), kc = " << kc() << ", acc = " << yAcc[i * kc() + k];
          ASSERT_EQ(
              uint32_t(yRef[i * kc() + k]), uint32_t(y[i * yStride() + k]))
              << "at pixel " << i << ", channel " << k << ", n = " << n()
              << ", ks = " << kh() << "x" << kw() << " (" << ks()
              << "), kc = " << kc() << ", acc = " << yAcc[i * kc() + k];
        }
      }
    }
  }

  void test(pytorch_q8avgpool_mp_ukernel_function q8avgpool) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<const uint8_t*> indirectX(packedKs() + (n() * s() - 1) * kh());
    std::vector<uint8_t> x((indirectX.size() - 1) * xStride() + kc());
    std::vector<int32_t, AlignedAllocator<int32_t, 16>> mpAcc(packedN());

    std::vector<uint8_t> zero(kc());
    std::vector<uint8_t> y((n() - 1) * yStride() + kc());
    std::vector<uint8_t> yRef(n() * kc());
    std::vector<float> yFP(n() * kc());
    std::vector<int32_t> yAcc(n() * kc());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      for (size_t i = 0; i < indirectX.size(); i++) {
        indirectX[i] = x.data() + i * xStride();
      }
      std::shuffle(indirectX.begin(), indirectX.end(), rng);

      /* Prepare quantization parameters */
      const union pytorch_qnnp_avgpool_quantization_params quantizationParams =
          pytorch_qnnp_compute_avgpool_quantization_params(
              -int32_t(xZeroPoint()) * int32_t(ks()),
              xScale() / (yScale() * float(ks())),
              yZeroPoint(),
              yMin(),
              yMax());
      const union pytorch_qnnp_avgpool_quantization_params
          scalarQuantizationParams =
              pytorch_qnnp_compute_scalar_avgpool_quantization_params(
                  -int32_t(xZeroPoint()) * int32_t(ks()),
                  xScale() / (yScale() * float(ks())),
                  yZeroPoint(),
                  yMin(),
                  yMax());

      /* Compute reference results */
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          int32_t acc = scalarQuantizationParams.scalar.bias;
          for (size_t j = 0; j < ks(); j++) {
            acc += indirectX[i * s() * kh() + j][k];
          }
          yAcc[i * kc() + k] = acc;
          yRef[i * kc() + k] =
              pytorch_qnnp_avgpool_quantize(acc, scalarQuantizationParams);
          yFP[i * kc() + k] =
              float(acc) * (xScale() / (yScale() * float(ks()))) +
              float(yZeroPoint());
          yFP[i * kc() + k] = std::min<float>(yFP[i * kc() + k], float(yMax()));
          yFP[i * kc() + k] = std::max<float>(yFP[i * kc() + k], float(yMin()));
        }
      }

      /* Call optimized micro-kernel */
      q8avgpool(
          n(),
          ks(),
          kc(),
          indirectX.data(),
          zero.data(),
          mpAcc.data(),
          y.data(),
          (kh() * s() - (packedKs() - qr())) * sizeof(void*),
          (yStride() - kc()) * sizeof(uint8_t),
          &quantizationParams);

      /* Verify results */
      for (size_t i = 0; i < n(); i++) {
        for (size_t k = 0; k < kc(); k++) {
          ASSERT_LE(uint32_t(y[i * yStride() + k]), uint32_t(yMax()))
              << "at pixel " << i << ", channel " << k << ", n = " << n()
              << ", kc = " << kc();
          ASSERT_GE(uint32_t(y[i * yStride() + k]), uint32_t(yMin()))
              << "at pixel " << i << ", channel " << k << ", n = " << n()
              << ", kc = " << kc();
          ASSERT_NEAR(
              float(int32_t(y[i * yStride() + k])), yFP[i * kc() + k], 0.5f)
              << "at pixel " << i << ", channel " << k << ", n = " << n()
              << ", ks = " << kh() << "x" << kw() << " (" << ks()
              << "), kc = " << kc() << ", acc = " << yAcc[i * kc() + k];
          ASSERT_EQ(
              uint32_t(yRef[i * kc() + k]), uint32_t(y[i * yStride() + k]))
              << "at pixel " << i << ", channel " << k << ", n = " << n()
              << ", ks = " << kh() << "x" << kw() << " (" << ks()
              << "), kc = " << kc() << ", acc = " << yAcc[i * kc() + k];
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
  float xScale_{1.25f};
  float yScale_{0.75f};
  uint8_t xZeroPoint_{121};
  uint8_t yZeroPoint_{133};
  uint8_t yMin_{0};
  uint8_t yMax_{255};
  size_t iterations_{15};
};
