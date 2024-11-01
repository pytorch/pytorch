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

#include <pytorch_qnnpack.h>

class SoftArgMaxOperatorTester {
 public:
  inline SoftArgMaxOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline SoftArgMaxOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride != 0);
    this->inputStride_ = inputStride;
    return *this;
  }

  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->inputStride_ >= this->channels_);
      return this->inputStride_;
    }
  }

  inline SoftArgMaxOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride != 0);
    this->outputStride_ = outputStride;
    return *this;
  }

  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return this->channels_;
    } else {
      assert(this->outputStride_ >= this->channels_);
      return this->outputStride_;
    }
  }

  inline SoftArgMaxOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline SoftArgMaxOperatorTester& inputScale(float inputScale) {
    assert(inputScale > 0.0f);
    assert(std::isnormal(inputScale));
    this->inputScale_ = inputScale;
    return *this;
  }

  inline float inputScale() const {
    return this->inputScale_;
  }

  inline SoftArgMaxOperatorTester& inputZeroPoint(uint8_t inputZeroPoint) {
    this->inputZeroPoint_ = inputZeroPoint;
    return *this;
  }

  inline uint8_t inputZeroPoint() const {
    return this->inputZeroPoint_;
  }

  inline float outputScale() const {
    return 1.0f / 256.0f;
  }

  inline uint8_t outputZeroPoint() const {
    return 0;
  }

  inline SoftArgMaxOperatorTester& iterations(size_t iterations) {
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

    std::vector<uint8_t> input((batchSize() - 1) * inputStride() + channels());
    std::vector<uint8_t> output(
        (batchSize() - 1) * outputStride() + channels());
    std::vector<float> outputRef(batchSize() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      /* Compute reference results */
      for (size_t i = 0; i < batchSize(); i++) {
        const int32_t maxInput = *std::max_element(
            input.data() + i * inputStride(),
            input.data() + i * inputStride() + channels());
        float sumExp = 0.0f;
        for (size_t c = 0; c < channels(); c++) {
          sumExp +=
              exp((int32_t(input[i * inputStride() + c]) - maxInput) *
                  inputScale());
        }
        for (size_t c = 0; c < channels(); c++) {
          outputRef[i * channels() + c] =
              exp((int32_t(input[i * inputStride() + c]) - maxInput) *
                  inputScale()) /
              (sumExp * outputScale());
          outputRef[i * channels() + c] =
              std::min(outputRef[i * channels() + c], 255.0f);
        }
      }

      /* Create, setup, run, and destroy SoftArgMax operator */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t softArgMaxOp = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_softargmax_nc_q8(
              channels(),
              inputScale(),
              outputZeroPoint(),
              outputScale(),
              0,
              &softArgMaxOp));
      ASSERT_NE(nullptr, softArgMaxOp);

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_softargmax_nc_q8(
              softArgMaxOp,
              batchSize(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(softArgMaxOp, nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_delete_operator(softArgMaxOp));
      softArgMaxOp = nullptr;

      /* Verify results */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_NEAR(
              float(int32_t(output[i * outputStride() + c])),
              outputRef[i * channels() + c],
              0.6f);
        }
      }
    }
  }

 private:
  size_t batchSize_{1};
  size_t channels_{1};
  size_t inputStride_{0};
  size_t outputStride_{0};
  float inputScale_{0.176080093};
  uint8_t inputZeroPoint_{121};
  size_t iterations_{15};
};
