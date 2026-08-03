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

class AveragePoolingOperatorTester {
 public:
  inline AveragePoolingOperatorTester& padding(uint32_t padding) {
    this->paddingHeight_ = padding;
    this->paddingWidth_ = padding;
    return *this;
  }

  inline AveragePoolingOperatorTester& padding(
      uint32_t paddingHeight,
      uint32_t paddingWidth) {
    this->paddingHeight_ = paddingHeight;
    this->paddingWidth_ = paddingWidth;
    return *this;
  }

  inline AveragePoolingOperatorTester& paddingHeight(uint32_t paddingHeight) {
    this->paddingHeight_ = paddingHeight;
    return *this;
  }

  inline AveragePoolingOperatorTester& paddingWidth(uint32_t paddingWidth) {
    this->paddingWidth_ = paddingWidth;
    return *this;
  }

  inline uint32_t paddingHeight() const {
    return this->paddingHeight_;
  }

  inline uint32_t paddingWidth() const {
    return this->paddingWidth_;
  }

  inline AveragePoolingOperatorTester& inputSize(
      size_t inputHeight,
      size_t inputWidth) {
    assert(inputHeight >= 1);
    assert(inputWidth >= 1);
    this->inputHeight_ = inputHeight;
    this->inputWidth_ = inputWidth;
    return *this;
  }

  inline AveragePoolingOperatorTester& inputHeight(size_t inputHeight) {
    assert(inputHeight >= 1);
    this->inputHeight_ = inputHeight;
    return *this;
  }

  inline size_t inputHeight() const {
    return this->inputHeight_;
  }

  inline AveragePoolingOperatorTester& inputWidth(size_t inputWidth) {
    assert(inputWidth >= 1);
    this->inputWidth_ = inputWidth;
    return *this;
  }

  inline size_t inputWidth() const {
    return this->inputWidth_;
  }

  inline AveragePoolingOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline AveragePoolingOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline AveragePoolingOperatorTester& poolingSize(uint32_t poolingSize) {
    assert(poolingSize >= 1);
    this->poolingHeight_ = poolingSize;
    this->poolingWidth_ = poolingSize;
    return *this;
  }

  inline AveragePoolingOperatorTester& poolingSize(
      uint32_t poolingHeight,
      uint32_t poolingWidth) {
    assert(poolingHeight >= 1);
    assert(poolingWidth >= 1);
    this->poolingHeight_ = poolingHeight;
    this->poolingWidth_ = poolingWidth;
    return *this;
  }

  inline AveragePoolingOperatorTester& poolingHeight(uint32_t poolingHeight) {
    assert(poolingHeight >= 1);
    this->poolingHeight_ = poolingHeight;
    return *this;
  }

  inline uint32_t poolingHeight() const {
    return this->poolingHeight_;
  }

  inline AveragePoolingOperatorTester& poolingWidth(uint32_t poolingWidth) {
    assert(poolingWidth >= 1);
    this->poolingWidth_ = poolingWidth;
    return *this;
  }

  inline uint32_t poolingWidth() const {
    return this->poolingWidth_;
  }

  inline AveragePoolingOperatorTester& stride(uint32_t stride) {
    assert(stride >= 1);
    this->strideHeight_ = stride;
    this->strideWidth_ = stride;
    return *this;
  }

  inline AveragePoolingOperatorTester& stride(
      uint32_t strideHeight,
      uint32_t strideWidth) {
    assert(strideHeight >= 1);
    assert(strideWidth >= 1);
    this->strideHeight_ = strideHeight;
    this->strideWidth_ = strideWidth;
    return *this;
  }

  inline AveragePoolingOperatorTester& strideHeight(uint32_t strideHeight) {
    assert(strideHeight >= 1);
    this->strideHeight_ = strideHeight;
    return *this;
  }

  inline uint32_t strideHeight() const {
    return this->strideHeight_;
  }

  inline AveragePoolingOperatorTester& strideWidth(uint32_t strideWidth) {
    assert(strideWidth >= 1);
    this->strideWidth_ = strideWidth;
    return *this;
  }

  inline uint32_t strideWidth() const {
    return this->strideWidth_;
  }

  inline size_t outputHeight() const {
    const size_t paddedInputHeight = inputHeight() + paddingHeight() * 2;
    if (paddedInputHeight <= poolingHeight()) {
      return 1;
    } else {
      return (paddedInputHeight - poolingHeight()) / strideHeight() + 1;
    }
  }

  inline size_t outputWidth() const {
    const size_t paddedInputWidth = inputWidth() + paddingWidth() * 2;
    if (paddedInputWidth <= poolingWidth()) {
      return 1;
    } else {
      return (paddedInputWidth - poolingWidth()) / strideWidth() + 1;
    }
  }

  inline AveragePoolingOperatorTester& inputPixelStride(
      size_t inputPixelStride) {
    assert(inputPixelStride != 0);
    this->inputPixelStride_ = inputPixelStride;
    return *this;
  }

  inline size_t inputPixelStride() const {
    if (this->inputPixelStride_ == 0) {
      return channels();
    } else {
      assert(this->inputPixelStride_ >= channels());
      return this->inputPixelStride_;
    }
  }

  inline AveragePoolingOperatorTester& outputPixelStride(
      size_t outputPixelStride) {
    assert(outputPixelStride != 0);
    this->outputPixelStride_ = outputPixelStride;
    return *this;
  }

  inline size_t outputPixelStride() const {
    if (this->outputPixelStride_ == 0) {
      return channels();
    } else {
      assert(this->outputPixelStride_ >= channels());
      return this->outputPixelStride_;
    }
  }

  inline AveragePoolingOperatorTester& nextInputSize(
      uint32_t nextInputHeight,
      uint32_t nextInputWidth) {
    assert(nextInputHeight >= 1);
    assert(nextInputWidth >= 1);
    this->nextInputHeight_ = nextInputHeight;
    this->nextInputWidth_ = nextInputWidth;
    return *this;
  }

  inline AveragePoolingOperatorTester& nextInputHeight(
      uint32_t nextInputHeight) {
    assert(nextInputHeight >= 1);
    this->nextInputHeight_ = nextInputHeight;
    return *this;
  }

  inline uint32_t nextInputHeight() const {
    if (this->nextInputHeight_ == 0) {
      return inputHeight();
    } else {
      return this->nextInputHeight_;
    }
  }

  inline AveragePoolingOperatorTester& nextInputWidth(uint32_t nextInputWidth) {
    assert(nextInputWidth >= 1);
    this->nextInputWidth_ = nextInputWidth;
    return *this;
  }

  inline uint32_t nextInputWidth() const {
    if (this->nextInputWidth_ == 0) {
      return inputWidth();
    } else {
      return this->nextInputWidth_;
    }
  }

  inline size_t nextOutputHeight() const {
    const size_t paddedNextInputHeight =
        nextInputHeight() + paddingHeight() * 2;
    if (paddedNextInputHeight <= poolingHeight()) {
      return 1;
    } else {
      return (paddedNextInputHeight - poolingHeight()) / strideHeight() + 1;
    }
  }

  inline size_t nextOutputWidth() const {
    const size_t paddedNextInputWidth = nextInputWidth() + paddingWidth() * 2;
    if (paddedNextInputWidth <= poolingWidth()) {
      return 1;
    } else {
      return (paddedNextInputWidth - poolingWidth()) / strideWidth() + 1;
    }
  }

  inline AveragePoolingOperatorTester& nextBatchSize(size_t nextBatchSize) {
    assert(nextBatchSize >= 1);
    this->nextBatchSize_ = nextBatchSize;
    return *this;
  }

  inline size_t nextBatchSize() const {
    if (this->nextBatchSize_ == 0) {
      return batchSize();
    } else {
      return this->nextBatchSize_;
    }
  }

  inline AveragePoolingOperatorTester& inputScale(float inputScale) {
    assert(inputScale > 0.0f);
    assert(std::isnormal(inputScale));
    this->inputScale_ = inputScale;
    return *this;
  }

  inline float inputScale() const {
    return this->inputScale_;
  }

  inline AveragePoolingOperatorTester& inputZeroPoint(uint8_t inputZeroPoint) {
    this->inputZeroPoint_ = inputZeroPoint;
    return *this;
  }

  inline uint8_t inputZeroPoint() const {
    return this->inputZeroPoint_;
  }

  inline AveragePoolingOperatorTester& outputScale(float outputScale) {
    assert(outputScale > 0.0f);
    assert(std::isnormal(outputScale));
    this->outputScale_ = outputScale;
    return *this;
  }

  inline float outputScale() const {
    return this->outputScale_;
  }

  inline AveragePoolingOperatorTester& outputZeroPoint(
      uint8_t outputZeroPoint) {
    this->outputZeroPoint_ = outputZeroPoint;
    return *this;
  }

  inline uint8_t outputZeroPoint() const {
    return this->outputZeroPoint_;
  }

  inline AveragePoolingOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline AveragePoolingOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline AveragePoolingOperatorTester& iterations(size_t iterations) {
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
        (batchSize() * inputHeight() * inputWidth() - 1) * inputPixelStride() +
        channels());
    std::vector<uint8_t> output(
        (batchSize() * outputHeight() * outputWidth() - 1) *
            outputPixelStride() +
        channels());
    std::vector<float> outputRef(
        batchSize() * outputHeight() * outputWidth() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      /* Compute reference results */
      const double scale = double(inputScale()) /
          (double(outputScale()) * double(poolingHeight() * poolingWidth()));
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t oy = 0; oy < outputHeight(); oy++) {
          for (size_t ox = 0; ox < outputWidth(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              double acc = 0.0f;
              for (size_t py = 0; py < poolingHeight(); py++) {
                const size_t iy = oy * strideHeight() + py - paddingHeight();
                for (size_t px = 0; px < poolingWidth(); px++) {
                  const size_t ix = ox * strideWidth() + px - paddingWidth();
                  if (ix < inputWidth() && iy < inputHeight()) {
                    acc += double(
                        int32_t(input
                                    [((i * inputHeight() + iy) * inputWidth() +
                                      ix) *
                                         inputPixelStride() +
                                     c]) -
                        int32_t(inputZeroPoint()));
                  }
                }
              }
              outputRef
                  [((i * outputHeight() + oy) * outputWidth() + ox) *
                       channels() +
                   c] = float(acc * scale + double(outputZeroPoint()));
              outputRef
                  [((i * outputHeight() + oy) * outputWidth() + ox) *
                       channels() +
                   c] =
                      std::min<float>(
                          outputRef
                              [((i * outputHeight() + oy) * outputWidth() +
                                ox) *
                                   channels() +
                               c],
                          float(qmax()));
              outputRef
                  [((i * outputHeight() + oy) * outputWidth() + ox) *
                       channels() +
                   c] =
                      std::max<float>(
                          outputRef
                              [((i * outputHeight() + oy) * outputWidth() +
                                ox) *
                                   channels() +
                               c],
                          float(qmin()));
            }
          }
        }
      }

      /* Create, setup, run, and destroy Average Pooling operator */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t averagePoolingOp = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_average_pooling2d_nhwc_q8(
              paddingHeight(),
              paddingWidth(),
              poolingHeight(),
              poolingWidth(),
              strideHeight(),
              strideWidth(),
              channels(),
              inputZeroPoint(),
              inputScale(),
              outputZeroPoint(),
              outputScale(),
              qmin(),
              qmax(),
              0,
              &averagePoolingOp));
      ASSERT_NE(nullptr, averagePoolingOp);

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_average_pooling2d_nhwc_q8(
              averagePoolingOp,
              batchSize(),
              inputHeight(),
              inputWidth(),
              input.data(),
              inputPixelStride(),
              output.data(),
              outputPixelStride(),
              nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(
              averagePoolingOp, nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_delete_operator(averagePoolingOp));
      averagePoolingOp = nullptr;

      /* Verify results */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t y = 0; y < outputHeight(); y++) {
          for (size_t x = 0; x < outputWidth(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_LE(
                  uint32_t(output
                               [((i * outputHeight() + y) * outputWidth() + x) *
                                    outputPixelStride() +
                                c]),
                  uint32_t(qmax()));
              ASSERT_GE(
                  uint32_t(output
                               [((i * outputHeight() + y) * outputWidth() + x) *
                                    outputPixelStride() +
                                c]),
                  uint32_t(qmin()));
              ASSERT_NEAR(
                  float(int32_t(
                      output
                          [((i * outputHeight() + y) * outputWidth() + x) *
                               outputPixelStride() +
                           c])),
                  outputRef
                      [((i * outputHeight() + y) * outputWidth() + x) *
                           channels() +
                       c],
                  0.80f)
                  << "in batch index " << i << ", pixel (" << y << ", " << x
                  << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void testSetupQ8() const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> input(std::max(
        (batchSize() * inputHeight() * inputWidth() - 1) * inputPixelStride() +
            channels(),
        (nextBatchSize() * nextInputHeight() * nextInputWidth() - 1) *
                inputPixelStride() +
            channels()));
    std::vector<uint8_t> output(std::max(
        (batchSize() * outputHeight() * outputWidth() - 1) *
                outputPixelStride() +
            channels(),
        (nextBatchSize() * nextOutputHeight() * nextOutputWidth() - 1) *
                outputPixelStride() +
            channels()));
    std::vector<float> outputRef(
        batchSize() * outputHeight() * outputWidth() * channels());
    std::vector<float> nextOutputRef(
        nextBatchSize() * nextOutputHeight() * nextOutputWidth() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      /* Compute reference results */
      const double scale = double(inputScale()) /
          (double(outputScale()) * double(poolingHeight() * poolingWidth()));
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t oy = 0; oy < outputHeight(); oy++) {
          for (size_t ox = 0; ox < outputWidth(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              double acc = 0.0f;
              for (size_t py = 0; py < poolingHeight(); py++) {
                const size_t iy = oy * strideHeight() + py - paddingHeight();
                for (size_t px = 0; px < poolingWidth(); px++) {
                  const size_t ix = ox * strideWidth() + px - paddingWidth();
                  if (ix < inputWidth() && iy < inputHeight()) {
                    acc += double(
                        int32_t(input
                                    [((i * inputHeight() + iy) * inputWidth() +
                                      ix) *
                                         inputPixelStride() +
                                     c]) -
                        int32_t(inputZeroPoint()));
                  }
                }
              }
              outputRef
                  [((i * outputHeight() + oy) * outputWidth() + ox) *
                       channels() +
                   c] = float(acc * scale + double(outputZeroPoint()));
              outputRef
                  [((i * outputHeight() + oy) * outputWidth() + ox) *
                       channels() +
                   c] =
                      std::min<float>(
                          outputRef
                              [((i * outputHeight() + oy) * outputWidth() +
                                ox) *
                                   channels() +
                               c],
                          float(qmax()));
              outputRef
                  [((i * outputHeight() + oy) * outputWidth() + ox) *
                       channels() +
                   c] =
                      std::max<float>(
                          outputRef
                              [((i * outputHeight() + oy) * outputWidth() +
                                ox) *
                                   channels() +
                               c],
                          float(qmin()));
            }
          }
        }
      }

      /* Create, setup, and run Average Pooling operator once */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t averagePoolingOp = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_average_pooling2d_nhwc_q8(
              paddingHeight(),
              paddingWidth(),
              poolingHeight(),
              poolingWidth(),
              strideHeight(),
              strideWidth(),
              channels(),
              inputZeroPoint(),
              inputScale(),
              outputZeroPoint(),
              outputScale(),
              qmin(),
              qmax(),
              0,
              &averagePoolingOp));
      ASSERT_NE(nullptr, averagePoolingOp);

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_average_pooling2d_nhwc_q8(
              averagePoolingOp,
              batchSize(),
              inputHeight(),
              inputWidth(),
              input.data(),
              inputPixelStride(),
              output.data(),
              outputPixelStride(),
              nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(
              averagePoolingOp, nullptr /* thread pool */));

      /* Verify results of the first run */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t y = 0; y < outputHeight(); y++) {
          for (size_t x = 0; x < outputWidth(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_LE(
                  uint32_t(output
                               [((i * outputHeight() + y) * outputWidth() + x) *
                                    outputPixelStride() +
                                c]),
                  uint32_t(qmax()));
              ASSERT_GE(
                  uint32_t(output
                               [((i * outputHeight() + y) * outputWidth() + x) *
                                    outputPixelStride() +
                                c]),
                  uint32_t(qmin()));
              ASSERT_NEAR(
                  float(int32_t(
                      output
                          [((i * outputHeight() + y) * outputWidth() + x) *
                               outputPixelStride() +
                           c])),
                  outputRef
                      [((i * outputHeight() + y) * outputWidth() + x) *
                           channels() +
                       c],
                  0.80f)
                  << "in batch index " << i << ", pixel (" << y << ", " << x
                  << "), channel " << c;
            }
          }
        }
      }

      /* Re-generate data for the second run */
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      /* Compute reference results for the second run */
      for (size_t i = 0; i < nextBatchSize(); i++) {
        for (size_t oy = 0; oy < nextOutputHeight(); oy++) {
          for (size_t ox = 0; ox < nextOutputWidth(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              double acc = 0.0f;
              for (size_t py = 0; py < poolingHeight(); py++) {
                const size_t iy = oy * strideHeight() + py - paddingHeight();
                for (size_t px = 0; px < poolingWidth(); px++) {
                  const size_t ix = ox * strideWidth() + px - paddingWidth();
                  if (ix < nextInputWidth() && iy < nextInputHeight()) {
                    acc += double(
                        int32_t(input
                                    [((i * nextInputHeight() + iy) *
                                          nextInputWidth() +
                                      ix) *
                                         inputPixelStride() +
                                     c]) -
                        int32_t(inputZeroPoint()));
                  }
                }
              }
              nextOutputRef
                  [((i * nextOutputHeight() + oy) * nextOutputWidth() + ox) *
                       channels() +
                   c] = float(acc * scale + double(outputZeroPoint()));
              nextOutputRef
                  [((i * nextOutputHeight() + oy) * nextOutputWidth() + ox) *
                       channels() +
                   c] =
                      std::min<float>(
                          nextOutputRef
                              [((i * nextOutputHeight() + oy) *
                                    nextOutputWidth() +
                                ox) *
                                   channels() +
                               c],
                          float(qmax()));
              nextOutputRef
                  [((i * nextOutputHeight() + oy) * nextOutputWidth() + ox) *
                       channels() +
                   c] =
                      std::max<float>(
                          nextOutputRef
                              [((i * nextOutputHeight() + oy) *
                                    nextOutputWidth() +
                                ox) *
                                   channels() +
                               c],
                          float(qmin()));
            }
          }
        }
      }

      /* Setup and run Average Pooling operator the second time, and destroy the
       * operator */
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_average_pooling2d_nhwc_q8(
              averagePoolingOp,
              nextBatchSize(),
              nextInputHeight(),
              nextInputWidth(),
              input.data(),
              inputPixelStride(),
              output.data(),
              outputPixelStride(),
              nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_run_operator(
              averagePoolingOp, nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_delete_operator(averagePoolingOp));
      averagePoolingOp = nullptr;

      /* Verify results of the second run */
      for (size_t i = 0; i < nextBatchSize(); i++) {
        for (size_t y = 0; y < nextOutputHeight(); y++) {
          for (size_t x = 0; x < nextOutputWidth(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_LE(
                  uint32_t(
                      output
                          [((i * nextOutputHeight() + y) * nextOutputWidth() +
                            x) *
                               outputPixelStride() +
                           c]),
                  uint32_t(qmax()));
              ASSERT_GE(
                  uint32_t(
                      output
                          [((i * nextOutputHeight() + y) * nextOutputWidth() +
                            x) *
                               outputPixelStride() +
                           c]),
                  uint32_t(qmin()));
              ASSERT_NEAR(
                  float(int32_t(
                      output
                          [((i * nextOutputHeight() + y) * nextOutputWidth() +
                            x) *
                               outputPixelStride() +
                           c])),
                  nextOutputRef
                      [((i * nextOutputHeight() + y) * nextOutputWidth() + x) *
                           channels() +
                       c],
                  0.80f)
                  << "in batch index " << i << ", pixel (" << y << ", " << x
                  << "), channel " << c;
            }
          }
        }
      }
    }
  }

 private:
  uint32_t paddingHeight_{0};
  uint32_t paddingWidth_{0};
  size_t inputHeight_{1};
  size_t inputWidth_{1};
  size_t channels_{1};
  size_t batchSize_{1};
  size_t inputPixelStride_{0};
  size_t outputPixelStride_{0};
  uint32_t poolingHeight_{1};
  uint32_t poolingWidth_{1};
  uint32_t strideHeight_{1};
  uint32_t strideWidth_{1};
  size_t nextInputHeight_{0};
  size_t nextInputWidth_{0};
  size_t nextBatchSize_{0};
  float inputScale_{1.0f};
  float outputScale_{1.0f};
  uint8_t inputZeroPoint_{121};
  uint8_t outputZeroPoint_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
