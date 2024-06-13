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

class GlobalAveragePoolingOperatorTester {
 public:
  inline GlobalAveragePoolingOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline GlobalAveragePoolingOperatorTester& width(size_t width) {
    assert(width != 0);
    this->width_ = width;
    return *this;
  }

  inline size_t width() const {
    return this->width_;
  }

  inline GlobalAveragePoolingOperatorTester& inputStride(size_t inputStride) {
    assert(inputStride != 0);
    this->inputStride_ = inputStride;
    return *this;
  }

  inline size_t inputStride() const {
    if (this->inputStride_ == 0) {
      return channels();
    } else {
      assert(this->inputStride_ >= channels());
      return this->inputStride_;
    }
  }

  inline GlobalAveragePoolingOperatorTester& outputStride(size_t outputStride) {
    assert(outputStride != 0);
    this->outputStride_ = outputStride;
    return *this;
  }

  inline size_t outputStride() const {
    if (this->outputStride_ == 0) {
      return channels();
    } else {
      assert(this->outputStride_ >= channels());
      return this->outputStride_;
    }
  }

  inline GlobalAveragePoolingOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline GlobalAveragePoolingOperatorTester& inputScale(float inputScale) {
    assert(inputScale > 0.0f);
    assert(std::isnormal(inputScale));
    this->inputScale_ = inputScale;
    return *this;
  }

  inline float inputScale() const {
    return this->inputScale_;
  }

  inline GlobalAveragePoolingOperatorTester& inputZeroPoint(
      uint8_t inputZeroPoint) {
    this->inputZeroPoint_ = inputZeroPoint;
    return *this;
  }

  inline uint8_t inputZeroPoint() const {
    return this->inputZeroPoint_;
  }

  inline GlobalAveragePoolingOperatorTester& outputScale(float outputScale) {
    assert(outputScale > 0.0f);
    assert(std::isnormal(outputScale));
    this->outputScale_ = outputScale;
    return *this;
  }

  inline float outputScale() const {
    return this->outputScale_;
  }

  inline GlobalAveragePoolingOperatorTester& outputZeroPoint(
      uint8_t outputZeroPoint) {
    this->outputZeroPoint_ = outputZeroPoint;
    return *this;
  }

  inline uint8_t outputZeroPoint() const {
    return this->outputZeroPoint_;
  }

  inline GlobalAveragePoolingOperatorTester& outputMin(uint8_t outputMin) {
    this->outputMin_ = outputMin;
    return *this;
  }

  inline uint8_t outputMin() const {
    return this->outputMin_;
  }

  inline GlobalAveragePoolingOperatorTester& outputMax(uint8_t outputMax) {
    this->outputMax_ = outputMax;
    return *this;
  }

  inline uint8_t outputMax() const {
    return this->outputMax_;
  }

  inline GlobalAveragePoolingOperatorTester& iterations(size_t iterations) {
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

    std::vector<uint8_t> input(
        (batchSize() * width() - 1) * inputStride() + channels());
    std::vector<uint8_t> output(batchSize() * outputStride());
    std::vector<float> outputRef(batchSize() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      /* Compute reference results */
      const double scale =
          double(inputScale()) / (double(width()) * double(outputScale()));
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          double acc = 0.0f;
          for (size_t k = 0; k < width(); k++) {
            acc += double(
                int32_t(input[(i * width() + k) * inputStride() + j]) -
                int32_t(inputZeroPoint()));
          }
          outputRef[i * channels() + j] =
              float(acc * scale + double(outputZeroPoint()));
          outputRef[i * channels() + j] = std::min<float>(
              outputRef[i * channels() + j], float(outputMax()));
          outputRef[i * channels() + j] = std::max<float>(
              outputRef[i * channels() + j], float(outputMin()));
        }
      }

      /* Create, setup, run, and destroy Add operator */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t globalAveragePoolingOp = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_global_average_pooling_nwc_q8(
              channels(),
              inputZeroPoint(),
              inputScale(),
              outputZeroPoint(),
              outputScale(),
              outputMin(),
              outputMax(),
              0,
              &globalAveragePoolingOp));
      ASSERT_NE(nullptr, globalAveragePoolingOp);

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_global_average_pooling_nwc_q8(
              globalAveragePoolingOp,
              batchSize(),
              width(),
              input.data(),
              inputStride(),
              output.data(),
              outputStride()));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(
              globalAveragePoolingOp, nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_delete_operator(globalAveragePoolingOp));
      globalAveragePoolingOp = nullptr;

      /* Verify results */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_LE(
              uint32_t(output[i * outputStride() + c]), uint32_t(outputMax()));
          ASSERT_GE(
              uint32_t(output[i * outputStride() + c]), uint32_t(outputMin()));
          ASSERT_NEAR(
              float(int32_t(output[i * outputStride() + c])),
              outputRef[i * channels() + c],
              0.80f)
              << "in batch index " << i << ", channel " << c;
        }
      }
    }
  }

 private:
  size_t batchSize_{1};
  size_t width_{1};
  size_t channels_{1};
  size_t inputStride_{0};
  size_t outputStride_{0};
  float inputScale_{1.0f};
  float outputScale_{1.0f};
  uint8_t inputZeroPoint_{121};
  uint8_t outputZeroPoint_{133};
  uint8_t outputMin_{0};
  uint8_t outputMax_{255};
  size_t iterations_{1};
};
