/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <pytorch_qnnpack.h>

class AddOperatorTester {
 public:
  inline AddOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline AddOperatorTester& aStride(size_t aStride) {
    assert(aStride != 0);
    this->aStride_ = aStride;
    return *this;
  }

  inline size_t aStride() const {
    if (this->aStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->aStride_ >= this->channels_);
      return this->aStride_;
    }
  }

  inline AddOperatorTester& bStride(size_t bStride) {
    assert(bStride != 0);
    this->bStride_ = bStride;
    return *this;
  }

  inline size_t bStride() const {
    if (this->bStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->bStride_ >= this->channels_);
      return this->bStride_;
    }
  }

  inline AddOperatorTester& yStride(size_t yStride) {
    assert(yStride != 0);
    this->yStride_ = yStride;
    return *this;
  }

  inline size_t yStride() const {
    if (this->yStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->yStride_ >= this->channels_);
      return this->yStride_;
    }
  }

  inline AddOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline AddOperatorTester& aScale(float aScale) {
    assert(aScale > 0.0f);
    assert(std::isnormal(aScale));
    this->aScale_ = aScale;
    return *this;
  }

  inline float aScale() const {
    return this->aScale_;
  }

  inline AddOperatorTester& aZeroPoint(uint8_t aZeroPoint) {
    this->aZeroPoint_ = aZeroPoint;
    return *this;
  }

  inline uint8_t aZeroPoint() const {
    return this->aZeroPoint_;
  }

  inline AddOperatorTester& bScale(float bScale) {
    assert(bScale > 0.0f);
    assert(std::isnormal(bScale));
    this->bScale_ = bScale;
    return *this;
  }

  inline float bScale() const {
    return this->bScale_;
  }

  inline AddOperatorTester& bZeroPoint(uint8_t bZeroPoint) {
    this->bZeroPoint_ = bZeroPoint;
    return *this;
  }

  inline uint8_t bZeroPoint() const {
    return this->bZeroPoint_;
  }

  inline AddOperatorTester& yScale(float yScale) {
    assert(yScale > 0.0f);
    assert(std::isnormal(yScale));
    this->yScale_ = yScale;
    return *this;
  }

  inline float yScale() const {
    return this->yScale_;
  }

  inline AddOperatorTester& yZeroPoint(uint8_t yZeroPoint) {
    this->yZeroPoint_ = yZeroPoint;
    return *this;
  }

  inline uint8_t yZeroPoint() const {
    return this->yZeroPoint_;
  }

  inline AddOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline AddOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline AddOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void testQ8() const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> a((batchSize() - 1) * aStride() + channels());
    std::vector<uint8_t> b((batchSize() - 1) * bStride() + channels());
    std::vector<uint8_t> y((batchSize() - 1) * yStride() + channels());
    std::vector<float> yRef(batchSize() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(a.begin(), a.end(), std::ref(u8rng));
      std::generate(b.begin(), b.end(), std::ref(u8rng));
      std::fill(y.begin(), y.end(), 0xA5);

      if (batchSize() * channels() > 3) {
        ASSERT_NE(
            *std::max_element(a.cbegin(), a.cend()),
            *std::min_element(a.cbegin(), a.cend()));
        ASSERT_NE(
            *std::max_element(b.cbegin(), b.cend()),
            *std::min_element(b.cbegin(), b.cend()));
      }

      /* Compute reference results */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          yRef[i * channels() + c] = float(yZeroPoint()) +
              float(int32_t(a[i * aStride() + c]) - int32_t(aZeroPoint())) *
                  (aScale() / yScale()) +
              float(int32_t(b[i * bStride() + c]) - int32_t(bZeroPoint())) *
                  (bScale() / yScale());
          yRef[i * channels() + c] =
              std::min<float>(yRef[i * channels() + c], float(qmax()));
          yRef[i * channels() + c] =
              std::max<float>(yRef[i * channels() + c], float(qmin()));
        }
      }

      /* Create, setup, run, and destroy Add operator */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t add_op = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_add_nc_q8(
              channels(),
              aZeroPoint(),
              aScale(),
              bZeroPoint(),
              bScale(),
              yZeroPoint(),
              yScale(),
              qmin(),
              qmax(),
              0,
              &add_op));
      ASSERT_NE(nullptr, add_op);

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_add_nc_q8(
              add_op,
              batchSize(),
              a.data(),
              aStride(),
              b.data(),
              bStride(),
              y.data(),
              yStride()));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(add_op, nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success, pytorch_qnnp_delete_operator(add_op));
      add_op = nullptr;

      /* Verify results */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(uint32_t(y[i * yStride() + c]), uint32_t(qmax()));
          ASSERT_GE(uint32_t(y[i * yStride() + c]), uint32_t(qmin()));
          ASSERT_NEAR(
              float(int32_t(y[i * yStride() + c])),
              yRef[i * channels() + c],
              0.6f);
        }
      }
    }
  }

 private:
  size_t batchSize_{1};
  size_t channels_{1};
  size_t aStride_{0};
  size_t bStride_{0};
  size_t yStride_{0};
  float aScale_{0.75f};
  float bScale_{1.25f};
  float yScale_{0.96875f};
  uint8_t aZeroPoint_{121};
  uint8_t bZeroPoint_{127};
  uint8_t yZeroPoint_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
