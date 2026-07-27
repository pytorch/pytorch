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

class MaxPoolingOperatorTester {
 public:
  inline MaxPoolingOperatorTester& padding(uint32_t padding) {
    this->paddingHeight_ = padding;
    this->paddingWidth_ = padding;
    return *this;
  }

  inline MaxPoolingOperatorTester& padding(
      uint32_t paddingHeight,
      uint32_t paddingWidth) {
    this->paddingHeight_ = paddingHeight;
    this->paddingWidth_ = paddingWidth;
    return *this;
  }

  inline MaxPoolingOperatorTester& paddingHeight(uint32_t paddingHeight) {
    this->paddingHeight_ = paddingHeight;
    return *this;
  }

  inline MaxPoolingOperatorTester& paddingWidth(uint32_t paddingWidth) {
    this->paddingWidth_ = paddingWidth;
    return *this;
  }

  inline uint32_t paddingHeight() const {
    return this->paddingHeight_;
  }

  inline uint32_t paddingWidth() const {
    return this->paddingWidth_;
  }

  inline MaxPoolingOperatorTester& inputSize(
      size_t inputHeight,
      size_t inputWidth) {
    assert(inputHeight >= 1);
    assert(inputWidth >= 1);
    this->inputHeight_ = inputHeight;
    this->inputWidth_ = inputWidth;
    return *this;
  }

  inline MaxPoolingOperatorTester& inputHeight(size_t inputHeight) {
    assert(inputHeight >= 1);
    this->inputHeight_ = inputHeight;
    return *this;
  }

  inline size_t inputHeight() const {
    return this->inputHeight_;
  }

  inline MaxPoolingOperatorTester& inputWidth(size_t inputWidth) {
    assert(inputWidth >= 1);
    this->inputWidth_ = inputWidth;
    return *this;
  }

  inline size_t inputWidth() const {
    return this->inputWidth_;
  }

  inline MaxPoolingOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline MaxPoolingOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline MaxPoolingOperatorTester& poolingSize(uint32_t poolingSize) {
    assert(poolingSize >= 1);
    this->poolingHeight_ = poolingSize;
    this->poolingWidth_ = poolingSize;
    return *this;
  }

  inline MaxPoolingOperatorTester& poolingSize(
      uint32_t poolingHeight,
      uint32_t poolingWidth) {
    assert(poolingHeight >= 1);
    assert(poolingWidth >= 1);
    this->poolingHeight_ = poolingHeight;
    this->poolingWidth_ = poolingWidth;
    return *this;
  }

  inline MaxPoolingOperatorTester& poolingHeight(uint32_t poolingHeight) {
    assert(poolingHeight >= 1);
    this->poolingHeight_ = poolingHeight;
    return *this;
  }

  inline uint32_t poolingHeight() const {
    return this->poolingHeight_;
  }

  inline MaxPoolingOperatorTester& poolingWidth(uint32_t poolingWidth) {
    assert(poolingWidth >= 1);
    this->poolingWidth_ = poolingWidth;
    return *this;
  }

  inline uint32_t poolingWidth() const {
    return this->poolingWidth_;
  }

  inline MaxPoolingOperatorTester& stride(uint32_t stride) {
    assert(stride >= 1);
    this->strideHeight_ = stride;
    this->strideWidth_ = stride;
    return *this;
  }

  inline MaxPoolingOperatorTester& stride(
      uint32_t strideHeight,
      uint32_t strideWidth) {
    assert(strideHeight >= 1);
    assert(strideWidth >= 1);
    this->strideHeight_ = strideHeight;
    this->strideWidth_ = strideWidth;
    return *this;
  }

  inline MaxPoolingOperatorTester& strideHeight(uint32_t strideHeight) {
    assert(strideHeight >= 1);
    this->strideHeight_ = strideHeight;
    return *this;
  }

  inline uint32_t strideHeight() const {
    return this->strideHeight_;
  }

  inline MaxPoolingOperatorTester& strideWidth(uint32_t strideWidth) {
    assert(strideWidth >= 1);
    this->strideWidth_ = strideWidth;
    return *this;
  }

  inline uint32_t strideWidth() const {
    return this->strideWidth_;
  }

  inline MaxPoolingOperatorTester& dilation(uint32_t dilation) {
    assert(dilation >= 1);
    this->dilationHeight_ = dilation;
    this->dilationWidth_ = dilation;
    return *this;
  }

  inline MaxPoolingOperatorTester& dilation(
      uint32_t dilationHeight,
      uint32_t dilationWidth) {
    assert(dilationHeight >= 1);
    assert(dilationWidth >= 1);
    this->dilationHeight_ = dilationHeight;
    this->dilationWidth_ = dilationWidth;
    return *this;
  }

  inline MaxPoolingOperatorTester& dilationHeight(uint32_t dilationHeight) {
    assert(dilationHeight >= 1);
    this->dilationHeight_ = dilationHeight;
    return *this;
  }

  inline uint32_t dilationHeight() const {
    return this->dilationHeight_;
  }

  inline MaxPoolingOperatorTester& dilationWidth(uint32_t dilationWidth) {
    assert(dilationWidth >= 1);
    this->dilationWidth_ = dilationWidth;
    return *this;
  }

  inline uint32_t dilationWidth() const {
    return this->dilationWidth_;
  }

  inline uint32_t dilatedPoolingHeight() const {
    return (poolingHeight() - 1) * dilationHeight() + 1;
  }

  inline uint32_t dilatedPoolingWidth() const {
    return (poolingWidth() - 1) * dilationWidth() + 1;
  }

  inline size_t outputHeight() const {
    const size_t paddedInputHeight = inputHeight() + paddingHeight() * 2;
    if (paddedInputHeight <= dilatedPoolingHeight()) {
      return 1;
    } else {
      return (paddedInputHeight - dilatedPoolingHeight()) / strideHeight() + 1;
    }
  }

  inline size_t outputWidth() const {
    const size_t paddedInputWidth = inputWidth() + paddingWidth() * 2;
    if (paddedInputWidth <= dilatedPoolingWidth()) {
      return 1;
    } else {
      return (paddedInputWidth - dilatedPoolingWidth()) / strideWidth() + 1;
    }
  }

  inline MaxPoolingOperatorTester& inputPixelStride(size_t inputPixelStride) {
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

  inline MaxPoolingOperatorTester& outputPixelStride(size_t outputPixelStride) {
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

  inline MaxPoolingOperatorTester& nextInputSize(
      uint32_t nextInputHeight,
      uint32_t nextInputWidth) {
    assert(nextInputHeight >= 1);
    assert(nextInputWidth >= 1);
    this->nextInputHeight_ = nextInputHeight;
    this->nextInputWidth_ = nextInputWidth;
    return *this;
  }

  inline MaxPoolingOperatorTester& nextInputHeight(uint32_t nextInputHeight) {
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

  inline MaxPoolingOperatorTester& nextInputWidth(uint32_t nextInputWidth) {
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
    if (paddedNextInputHeight <= dilatedPoolingHeight()) {
      return 1;
    } else {
      return (paddedNextInputHeight - dilatedPoolingHeight()) / strideHeight() +
          1;
    }
  }

  inline size_t nextOutputWidth() const {
    const size_t paddedNextInputWidth = nextInputWidth() + paddingWidth() * 2;
    if (paddedNextInputWidth <= dilatedPoolingWidth()) {
      return 1;
    } else {
      return (paddedNextInputWidth - dilatedPoolingWidth()) / strideWidth() + 1;
    }
  }

  inline MaxPoolingOperatorTester& nextBatchSize(size_t nextBatchSize) {
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

  inline MaxPoolingOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline MaxPoolingOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline MaxPoolingOperatorTester& iterations(size_t iterations) {
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

    std::vector<uint8_t> input(
        (batchSize() * inputHeight() * inputWidth() - 1) * inputPixelStride() +
        channels());
    std::vector<uint8_t> output(
        (batchSize() * outputHeight() * outputWidth() - 1) *
            outputPixelStride() +
        channels());
    std::vector<uint8_t> outputRef(
        batchSize() * outputHeight() * outputWidth() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      /* Compute reference results */
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t oy = 0; oy < outputHeight(); oy++) {
          for (size_t ox = 0; ox < outputWidth(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              uint8_t maxValue = 0;
              for (size_t py = 0; py < poolingHeight(); py++) {
                const size_t iy = oy * strideHeight() + py * dilationHeight() -
                    paddingHeight();
                for (size_t px = 0; px < poolingWidth(); px++) {
                  const size_t ix = ox * strideWidth() + px * dilationWidth() -
                      paddingWidth();
                  if (ix < inputWidth() && iy < inputHeight()) {
                    maxValue = std::max(
                        maxValue,
                        input
                            [((i * inputHeight() + iy) * inputWidth() + ix) *
                                 inputPixelStride() +
                             c]);
                  }
                }
              }
              maxValue = std::min(maxValue, qmax());
              maxValue = std::max(maxValue, qmin());
              outputRef
                  [((i * outputHeight() + oy) * outputWidth() + ox) *
                       channels() +
                   c] = maxValue;
            }
          }
        }
      }

      /* Create, setup, run, and destroy Max Pooling operator */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t maxPoolingOp = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_max_pooling2d_nhwc_u8(
              paddingHeight(),
              paddingWidth(),
              poolingHeight(),
              poolingWidth(),
              strideHeight(),
              strideWidth(),
              dilationHeight(),
              dilationWidth(),
              channels(),
              qmin(),
              qmax(),
              0,
              &maxPoolingOp));
      ASSERT_NE(nullptr, maxPoolingOp);

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
              maxPoolingOp,
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
          pytorch_qnnp_run_operator(maxPoolingOp, nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_delete_operator(maxPoolingOp));
      maxPoolingOp = nullptr;

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
              ASSERT_EQ(
                  uint32_t(outputRef
                               [((i * outputHeight() + y) * outputWidth() + x) *
                                    channels() +
                                c]),
                  uint32_t(output
                               [((i * outputHeight() + y) * outputWidth() + x) *
                                    outputPixelStride() +
                                c]))
                  << "in batch index " << i << ", pixel (" << y << ", " << x
                  << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void testSetupU8() const {
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
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t oy = 0; oy < outputHeight(); oy++) {
          for (size_t ox = 0; ox < outputWidth(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              uint8_t maxValue = 0;
              for (size_t py = 0; py < poolingHeight(); py++) {
                const size_t iy = oy * strideHeight() + py * dilationHeight() -
                    paddingHeight();
                for (size_t px = 0; px < poolingWidth(); px++) {
                  const size_t ix = ox * strideWidth() + px * dilationWidth() -
                      paddingWidth();
                  if (ix < inputWidth() && iy < inputHeight()) {
                    maxValue = std::max(
                        maxValue,
                        input
                            [((i * inputHeight() + iy) * inputWidth() + ix) *
                                 inputPixelStride() +
                             c]);
                  }
                }
              }
              maxValue = std::min(maxValue, qmax());
              maxValue = std::max(maxValue, qmin());
              outputRef
                  [((i * outputHeight() + oy) * outputWidth() + ox) *
                       channels() +
                   c] = maxValue;
            }
          }
        }
      }

      /* Create, setup, and run Max Pooling operator once */
      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      pytorch_qnnp_operator_t maxPoolingOp = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_create_max_pooling2d_nhwc_u8(
              paddingHeight(),
              paddingWidth(),
              poolingHeight(),
              poolingWidth(),
              strideHeight(),
              strideWidth(),
              dilationHeight(),
              dilationWidth(),
              channels(),
              qmin(),
              qmax(),
              0,
              &maxPoolingOp));
      ASSERT_NE(nullptr, maxPoolingOp);

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
              maxPoolingOp,
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
          pytorch_qnnp_run_operator(maxPoolingOp, nullptr /* thread pool */));

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
              ASSERT_EQ(
                  uint32_t(outputRef
                               [((i * outputHeight() + y) * outputWidth() + x) *
                                    channels() +
                                c]),
                  uint32_t(output
                               [((i * outputHeight() + y) * outputWidth() + x) *
                                    outputPixelStride() +
                                c]))
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
              uint8_t maxValue = 0;
              for (size_t py = 0; py < poolingHeight(); py++) {
                const size_t iy = oy * strideHeight() + py * dilationHeight() -
                    paddingHeight();
                for (size_t px = 0; px < poolingWidth(); px++) {
                  const size_t ix = ox * strideWidth() + px * dilationWidth() -
                      paddingWidth();
                  if (ix < nextInputWidth() && iy < nextInputHeight()) {
                    maxValue = std::max(
                        maxValue,
                        input
                            [((i * nextInputHeight() + iy) * nextInputWidth() +
                              ix) *
                                 inputPixelStride() +
                             c]);
                  }
                }
              }
              maxValue = std::min(maxValue, qmax());
              maxValue = std::max(maxValue, qmin());
              nextOutputRef
                  [((i * nextOutputHeight() + oy) * nextOutputWidth() + ox) *
                       channels() +
                   c] = maxValue;
            }
          }
        }
      }

      /* Setup and run Max Pooling operator the second time, and destroy the
       * operator */
      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
              maxPoolingOp,
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
          pytorch_qnnp_run_operator(maxPoolingOp, nullptr /* thread pool */));

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          pytorch_qnnp_delete_operator(maxPoolingOp));
      maxPoolingOp = nullptr;

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
              ASSERT_EQ(
                  uint32_t(
                      nextOutputRef
                          [((i * nextOutputHeight() + y) * nextOutputWidth() +
                            x) *
                               channels() +
                           c]),
                  uint32_t(
                      output
                          [((i * nextOutputHeight() + y) * nextOutputWidth() +
                            x) *
                               outputPixelStride() +
                           c]))
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
  uint32_t dilationHeight_{1};
  uint32_t dilationWidth_{1};
  size_t nextInputHeight_{0};
  size_t nextInputWidth_{0};
  size_t nextBatchSize_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
