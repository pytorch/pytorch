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

#include <pytorch_qnnpack.h>

class ClampOperatorTester {
 public:
  inline ClampOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline ClampOperatorTester& inputStride(size_t inputStride) {
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

  inline ClampOperatorTester& outputStride(size_t outputStride) {
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

  inline ClampOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline ClampOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline ClampOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline ClampOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void testU8() const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> input((batchSize() - 1) * inputStride() + channels());
    std::vector<uint8_t> output(
        (batchSize() - 1) * outputStride() + channels());
    std::vector<uint8_t> outputRef(batchSize() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      /* Compute reference results */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          const uint8_t x = input[i * inputStride() + c];
          const uint8_t y = std::min(std::max(x, qmin()), qmax());
          outputRef[i * channels() + c] = y;
        }
      }

      /* Create, setup, run, and destroy Sigmoid operator */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t clampOp = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_clamp_nc_u8(
              channels(), qmin(), qmax(), 0, &clampOp));
      ASSERT_NE(nullptr, clampOp);

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_clamp_nc_u8(
              clampOp,
              batchSize(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(clampOp, nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success, pytorch_qnnp_delete_operator(clampOp));
      clampOp = nullptr;

      /* Verify results */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(uint32_t(output[i * channels() + c]), uint32_t(qmax()))
              << "at position " << i << ", batch size = " << batchSize()
              << ", channels = " << channels();
          ASSERT_GE(uint32_t(output[i * channels() + c]), uint32_t(qmin()))
              << "at position " << i << ", batch size = " << batchSize()
              << ", channels = " << channels();
          ASSERT_EQ(
              uint32_t(outputRef[i * channels() + c]),
              uint32_t(output[i * outputStride() + c]))
              << "at position " << i << ", batch size = " << batchSize()
              << ", channels = " << channels() << ", qmin = " << qmin()
              << ", qmax = " << qmax();
        }
      }
    }
  }

 private:
  size_t batchSize_{1};
  size_t channels_{1};
  size_t inputStride_{0};
  size_t outputStride_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
