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
#include <memory>

#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>

#include "test_utils.h"
using namespace qnnpack::testing;

class ConvolutionOperatorTester {
 public:
  inline size_t dimensionality() const {
    return this->dimensionality_;
  }

  inline ConvolutionOperatorTester& dimensionality(size_t dimensionality) {
    assert(dimensionality == 2 || dimensionality == 3);
    this->dimensionality_ = dimensionality;
    return *this;
  }

  inline ConvolutionOperatorTester& padding(uint32_t padding) {
    if (this->dimensionality_ == 3) {
      this->paddingDepth_ = padding;
    }
    this->paddingHeight_ = padding;
    this->paddingWidth_ = padding;
    return *this;
  }

  inline ConvolutionOperatorTester& padding(
      uint32_t paddingHeight,
      uint32_t paddingWidth) {
    this->paddingHeight_ = paddingHeight;
    this->paddingWidth_ = paddingWidth;
    return *this;
  }

  inline ConvolutionOperatorTester& padding(
      uint32_t paddingDepth,
      uint32_t paddingHeight,
      uint32_t paddingWidth) {
    this->paddingDepth_ = paddingDepth;
    return this->padding(paddingHeight, paddingWidth);
  }

  inline ConvolutionOperatorTester& paddingDepth(uint32_t paddingDepth) {
    this->paddingDepth_ = paddingDepth;
    return *this;
  }

  inline ConvolutionOperatorTester& paddingHeight(uint32_t paddingHeight) {
    this->paddingHeight_ = paddingHeight;
    return *this;
  }

  inline ConvolutionOperatorTester& paddingWidth(uint32_t paddingWidth) {
    this->paddingWidth_ = paddingWidth;
    return *this;
  }

  inline uint32_t paddingDepth() const {
    return this->paddingDepth_;
  }

  inline uint32_t paddingHeight() const {
    return this->paddingHeight_;
  }

  inline uint32_t paddingWidth() const {
    return this->paddingWidth_;
  }

  inline ConvolutionOperatorTester& inputSize(
      uint32_t inputHeight,
      uint32_t inputWidth) {
    assert(inputHeight >= 1);
    assert(inputWidth >= 1);
    this->inputHeight_ = inputHeight;
    this->inputWidth_ = inputWidth;
    return *this;
  }

  inline ConvolutionOperatorTester& inputSize(
      uint32_t inputDepth,
      uint32_t inputHeight,
      uint32_t inputWidth) {
    assert(inputDepth >= 1);
    this->inputDepth_ = inputDepth;
    return this->inputSize(inputHeight, inputWidth);
  }

  inline ConvolutionOperatorTester& inputDepth(uint32_t inputDepth) {
    assert(inputDepth >= 1);
    this->inputDepth_ = inputDepth;
    return *this;
  }

  inline uint32_t inputDepth() const {
    return this->inputDepth_;
  }

  inline ConvolutionOperatorTester& inputHeight(uint32_t inputHeight) {
    assert(inputHeight >= 1);
    this->inputHeight_ = inputHeight;
    return *this;
  }

  inline uint32_t inputHeight() const {
    return this->inputHeight_;
  }

  inline ConvolutionOperatorTester& inputWidth(uint32_t inputWidth) {
    assert(inputWidth >= 1);
    this->inputWidth_ = inputWidth;
    return *this;
  }

  inline uint32_t inputWidth() const {
    return this->inputWidth_;
  }

  inline ConvolutionOperatorTester& groups(uint32_t groups) {
    assert(groups >= 1);
    this->groups_ = groups;
    return *this;
  }

  inline uint32_t groups() const {
    return this->groups_;
  }

  inline ConvolutionOperatorTester& groupInputChannels(
      size_t groupInputChannels) {
    assert(groupInputChannels >= 1);
    this->groupInputChannels_ = groupInputChannels;
    return *this;
  }

  inline size_t groupInputChannels() const {
    return this->groupInputChannels_;
  }

  inline ConvolutionOperatorTester& per_channel(bool per_channel) {
    this->per_channel_ = per_channel;
    return *this;
  }

  inline bool per_channel() const {
    return this->per_channel_;
  }

  inline ConvolutionOperatorTester& groupOutputChannels(
      size_t groupOutputChannels) {
    assert(groupOutputChannels >= 1);
    this->groupOutputChannels_ = groupOutputChannels;
    return *this;
  }

  inline size_t groupOutputChannels() const {
    return this->groupOutputChannels_;
  }

  inline ConvolutionOperatorTester& batchSize(size_t batchSize) {
    this->batchSize_ = batchSize;
    return *this;
  }

  inline size_t batchSize() const {
    return this->batchSize_;
  }

  inline ConvolutionOperatorTester& kernelSize(uint32_t kernelSize) {
    assert(kernelSize >= 1);
    if (this->dimensionality_ == 3) {
      this->kernelDepth_ = kernelSize;
    }
    this->kernelHeight_ = kernelSize;
    this->kernelWidth_ = kernelSize;
    return *this;
  }

  inline ConvolutionOperatorTester& kernelSize(
      uint32_t kernelHeight,
      uint32_t kernelWidth) {
    assert(kernelHeight >= 1);
    assert(kernelWidth >= 1);
    this->kernelHeight_ = kernelHeight;
    this->kernelWidth_ = kernelWidth;
    return *this;
  }

  inline ConvolutionOperatorTester& kernelSize(
      uint32_t kernelDepth,
      uint32_t kernelHeight,
      uint32_t kernelWidth) {
    assert(kernelDepth >= 1);
    this->kernelDepth_ = kernelDepth;
    return this->kernelSize(kernelHeight, kernelWidth);
  }

  inline ConvolutionOperatorTester& kernelDepth(uint32_t kernelDepth) {
    assert(kernelDepth >= 1);
    this->kernelDepth_ = kernelDepth;
    return *this;
  }

  inline uint32_t kernelDepth() const {
    return this->kernelDepth_;
  }

  inline ConvolutionOperatorTester& kernelHeight(uint32_t kernelHeight) {
    assert(kernelHeight >= 1);
    this->kernelHeight_ = kernelHeight;
    return *this;
  }

  inline uint32_t kernelHeight() const {
    return this->kernelHeight_;
  }

  inline ConvolutionOperatorTester& kernelWidth(uint32_t kernelWidth) {
    assert(kernelWidth >= 1);
    this->kernelWidth_ = kernelWidth;
    return *this;
  }

  inline uint32_t kernelWidth() const {
    return this->kernelWidth_;
  }

  inline ConvolutionOperatorTester& dilation(uint32_t dilation) {
    assert(dilation >= 1);
    if (this->dimensionality_ == 3) {
      this->dilationDepth_ = dilation;
    }
    this->dilationHeight_ = dilation;
    this->dilationWidth_ = dilation;
    return *this;
  }

  inline ConvolutionOperatorTester& dilation(
      uint32_t dilationHeight,
      uint32_t dilationWidth) {
    assert(dilationHeight >= 1);
    assert(dilationWidth >= 1);
    this->dilationHeight_ = dilationHeight;
    this->dilationWidth_ = dilationWidth;
    return *this;
  }

  inline ConvolutionOperatorTester& dilation(
      uint32_t dilationDepth,
      uint32_t dilationHeight,
      uint32_t dilationWidth) {
    assert(dilationDepth >= 1);
    this->dilationDepth_ = dilationDepth;
    return this->dilation(dilationHeight, dilationWidth);
  }

  inline ConvolutionOperatorTester& dilationDepth(uint32_t dilationDepth) {
    assert(dilationDepth >= 1);
    this->dilationDepth_ = dilationDepth;
    return *this;
  }

  inline uint32_t dilationDepth() const {
    return this->dilationDepth_;
  }

  inline ConvolutionOperatorTester& dilationHeight(uint32_t dilationHeight) {
    assert(dilationHeight >= 1);
    this->dilationHeight_ = dilationHeight;
    return *this;
  }

  inline uint32_t dilationHeight() const {
    return this->dilationHeight_;
  }

  inline ConvolutionOperatorTester& dilationWidth(uint32_t dilationWidth) {
    assert(dilationWidth >= 1);
    this->dilationWidth_ = dilationWidth;
    return *this;
  }

  inline uint32_t dilationWidth() const {
    return this->dilationWidth_;
  }

  inline ConvolutionOperatorTester& subsampling(uint32_t subsampling) {
    assert(subsampling >= 1);
    if (this->dimensionality_ == 3) {
      this->subsamplingDepth_ = subsampling;
    }
    this->subsamplingHeight_ = subsampling;
    this->subsamplingWidth_ = subsampling;
    return *this;
  }

  inline ConvolutionOperatorTester& subsampling(
      uint32_t subsamplingHeight,
      uint32_t subsamplingWidth) {
    assert(subsamplingHeight >= 1);
    assert(subsamplingWidth >= 1);
    this->subsamplingHeight_ = subsamplingHeight;
    this->subsamplingWidth_ = subsamplingWidth;
    return *this;
  }

  inline ConvolutionOperatorTester& subsampling(
      uint32_t subsamplingDepth,
      uint32_t subsamplingHeight,
      uint32_t subsamplingWidth) {
    assert(subsamplingDepth >= 1);
    this->subsamplingDepth_ = subsamplingDepth;
    return this->subsampling(subsamplingHeight, subsamplingWidth);
  }

  inline ConvolutionOperatorTester& subsamplingDepth(
      uint32_t subsamplingDepth) {
    assert(subsamplingDepth >= 1);
    this->subsamplingDepth_ = subsamplingDepth;
    return *this;
  }

  inline uint32_t subsamplingDepth() const {
    return this->subsamplingDepth_;
  }

  inline ConvolutionOperatorTester& subsamplingHeight(
      uint32_t subsamplingHeight) {
    assert(subsamplingHeight >= 1);
    this->subsamplingHeight_ = subsamplingHeight;
    return *this;
  }

  inline uint32_t subsamplingHeight() const {
    return this->subsamplingHeight_;
  }

  inline ConvolutionOperatorTester& subsamplingWidth(
      uint32_t subsamplingWidth) {
    assert(subsamplingWidth >= 1);
    this->subsamplingWidth_ = subsamplingWidth;
    return *this;
  }

  inline uint32_t subsamplingWidth() const {
    return this->subsamplingWidth_;
  }

  inline ConvolutionOperatorTester& inputPixelStride(size_t inputPixelStride) {
    assert(inputPixelStride >= 1);
    this->inputPixelStride_ = inputPixelStride;
    return *this;
  }

  inline size_t inputPixelStride() const {
    if (this->inputPixelStride_ == 0) {
      return groupInputChannels() * groups();
    } else {
      assert(this->inputPixelStride_ >= groupInputChannels() * groups());
      return this->inputPixelStride_;
    }
  }

  inline ConvolutionOperatorTester& outputPixelStride(
      size_t outputPixelStride) {
    assert(outputPixelStride >= 1);
    this->outputPixelStride_ = outputPixelStride;
    return *this;
  }

  inline size_t outputPixelStride() const {
    if (this->outputPixelStride_ == 0) {
      return groupOutputChannels() * groups();
    } else {
      assert(this->outputPixelStride_ >= groupOutputChannels() * groups());
      return this->outputPixelStride_;
    }
  }

  inline uint32_t dilatedKernelDepth() const {
    return (kernelDepth() - 1) * dilationDepth() + 1;
  }

  inline uint32_t dilatedKernelHeight() const {
    return (kernelHeight() - 1) * dilationHeight() + 1;
  }

  inline uint32_t dilatedKernelWidth() const {
    return (kernelWidth() - 1) * dilationWidth() + 1;
  }

  inline size_t outputDepth() const {
    const size_t paddedInputDepth = inputDepth() + paddingDepth() * 2;
    if (paddedInputDepth <= dilatedKernelDepth()) {
      return 1;
    } else {
      return (paddedInputDepth - dilatedKernelDepth()) / subsamplingDepth() + 1;
    }
  }

  inline size_t outputHeight() const {
    const size_t paddedInputHeight = inputHeight() + paddingHeight() * 2;
    if (paddedInputHeight <= dilatedKernelHeight()) {
      return 1;
    } else {
      return (paddedInputHeight - dilatedKernelHeight()) / subsamplingHeight() +
          1;
    }
  }

  inline size_t outputWidth() const {
    const size_t paddedInputWidth = inputWidth() + paddingWidth() * 2;
    if (paddedInputWidth <= dilatedKernelWidth()) {
      return 1;
    } else {
      return (paddedInputWidth - dilatedKernelWidth()) / subsamplingWidth() + 1;
    }
  }

  inline ConvolutionOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline ConvolutionOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline ConvolutionOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void testQ8(const Mode mode = Mode::Static) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);
    auto f32rng =
        std::bind(std::uniform_real_distribution<float>(1, 5), rng);

    std::vector<uint8_t> input(
        batchSize() *
            ((inputDepth() * inputHeight() * inputWidth() - 1) *
                 inputPixelStride() +
             groups() * groupInputChannels()) +
        8);
    std::vector<uint8_t> kernel(
        groups() * groupOutputChannels() * kernelHeight() * kernelDepth() *
        kernelWidth() * groupInputChannels());
    std::vector<int32_t> bias(groups() * groupOutputChannels());
    std::vector<uint8_t> output(
        batchSize() *
        ((outputDepth() * outputHeight() * outputWidth() - 1) *
             outputPixelStride() +
         groups() * groupOutputChannels()));
    std::vector<int32_t> accumulators(
        batchSize() * outputDepth() * outputHeight() * outputWidth() *
        groups() * groupOutputChannels());

    const uint8_t* inputPtr = input.data() + 8;
    const uint8_t inputZeroPoint = 127;
    // Make num zero points multiple of 8.
    // This is the least common denominator for SSE/ARM kernels we have.
    size_t num_zero_points_padded =
      (groups() * groupOutputChannels() + 8);
    std::vector<uint8_t> kernelZeroPoints(num_zero_points_padded, 127);

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      if (per_channel()) {
        std::generate(kernelZeroPoints.begin(), kernelZeroPoints.end(), std::ref(u8rng));
      }
      std::fill(output.begin(), output.end(), 0xA5);
      std::fill(accumulators.begin(), accumulators.end(), 0);

      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t oz = 0; oz < outputDepth(); oz++) {
          for (size_t oy = 0; oy < outputHeight(); oy++) {
            for (size_t ox = 0; ox < outputWidth(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < groupOutputChannels(); oc++) {
                  accumulators
                      [((((i * outputDepth() + oz) * outputHeight() + oy) *
                             outputWidth() +
                         ox) *
                            groups() +
                        g) *
                           groupOutputChannels() +
                       oc] = bias[g * groupOutputChannels() + oc];
                }
              }
            }
          }
        }
      }
      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t oz = 0; oz < outputDepth(); oz++) {
          for (size_t oy = 0; oy < outputHeight(); oy++) {
            for (size_t ox = 0; ox < outputWidth(); ox++) {
              for (size_t kz = 0; kz < kernelDepth(); kz++) {
                const size_t iz = oz * subsamplingDepth() +
                    kz * dilationDepth() - paddingDepth();
                if (iz < inputDepth()) {
                  for (size_t ky = 0; ky < kernelHeight(); ky++) {
                    const size_t iy = oy * subsamplingHeight() +
                        ky * dilationHeight() - paddingHeight();
                    if (iy < inputHeight()) {
                      for (size_t kx = 0; kx < kernelWidth(); kx++) {
                        const size_t ix = ox * subsamplingWidth() +
                            kx * dilationWidth() - paddingWidth();
                        if (ix < inputWidth()) {
                          for (size_t g = 0; g < groups(); g++) {
                            for (size_t oc = 0; oc < groupOutputChannels();
                                 oc++) {
                              for (size_t ic = 0; ic < groupInputChannels();
                                   ic++) {
                                accumulators
                                    [((((i * outputDepth() + oz) *
                                            outputHeight() +
                                        oy) *
                                           outputWidth() +
                                       ox) *
                                          groups() +
                                      g) *
                                         groupOutputChannels() +
                                     oc] +=
                                    (int32_t(
                                         inputPtr
                                             [(((i * inputDepth() + iz) *
                                                    inputHeight() +
                                                iy) *
                                                   inputWidth() +
                                               ix) *
                                                  inputPixelStride() +
                                              g * groupInputChannels() + ic]) -
                                     int32_t(inputZeroPoint)) *
                                    (int32_t(
                                         kernel
                                             [((((g * groupOutputChannels() +
                                                  oc) *
                                                     kernelDepth() +
                                                 kz) *
                                                    kernelHeight() +
                                                ky) *
                                                   kernelWidth() +
                                               kx) *
                                                  groupInputChannels() +
                                              ic]) -
                                     int32_t(
                                         kernelZeroPoints
                                             [g * groupOutputChannels() + oc]));
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      // Create dummy min/max for empty inputs.
      // These are only used to compute scale and zero point,
      // and real callers will just pull those values from the model.
      const int32_t accumulatorsMin = accumulators.empty()
          ? 0
          : *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulatorsMax = accumulators.empty()
          ? 900
          : *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double outputScale =
          double(uint32_t(accumulatorsMax - accumulatorsMin)) / 255.0;
      const uint8_t outputZeroPoint = uint8_t(std::max(
          std::min(
              lrint(
                  127.5 -
                  0.5 * double(accumulatorsMin + accumulatorsMax) /
                      outputScale),
              long(std::numeric_limits<uint8_t>::max())),
          long(std::numeric_limits<uint8_t>::min())));

      ASSERT_EQ(pytorch_qnnp_status_success, pytorch_qnnp_initialize());
      std::vector<float> requantization_scales(num_zero_points_padded, 1.0 * 1.0 / outputScale);
      if (per_channel()) {
        auto scale_generator = [&]() -> float {return (f32rng()/outputScale);};
        std::generate(
            requantization_scales.begin(),
            requantization_scales.end(),
            std::ref(scale_generator));
      }

      pytorch_qnnp_operator_t convolution = nullptr;

      ASSERT_EQ(
          pytorch_qnnp_status_success,
          (dimensionality() == 2 ? pytorch_qnnp_create_convolution2d_nhwc_q8(
                                       paddingHeight(),
                                       paddingWidth(),
                                       kernelHeight(),
                                       kernelWidth(),
                                       subsamplingHeight(),
                                       subsamplingWidth(),
                                       dilationHeight(),
                                       dilationWidth(),
                                       groups(),
                                       groupInputChannels(),
                                       groupOutputChannels(),
                                       inputZeroPoint,
                                       kernelZeroPoints.data(),
                                       kernel.data(),
                                       bias.data(),
                                       outputZeroPoint,
                                       qmin(),
                                       qmax(),
                                       0,
                                       requantization_scales.data(),
                                       per_channel(),
                                       &convolution)
                                 : pytorch_qnnp_create_convolution3d_ndhwc_q8(
                                       paddingDepth(),
                                       paddingHeight(),
                                       paddingWidth(),
                                       kernelDepth(),
                                       kernelHeight(),
                                       kernelWidth(),
                                       subsamplingDepth(),
                                       subsamplingHeight(),
                                       subsamplingWidth(),
                                       dilationDepth(),
                                       dilationHeight(),
                                       dilationWidth(),
                                       groups(),
                                       groupInputChannels(),
                                       groupOutputChannels(),
                                       inputZeroPoint,
                                       kernelZeroPoints.data(),
                                       kernel.data(),
                                       bias.data(),
                                       outputZeroPoint,
                                       qmin(),
                                       qmax(),
                                       0,
                                       requantization_scales.data(),
                                       per_channel(),
                                       &convolution)));
      switch (mode) {
        case Mode::Static: {
          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_setup_convolution_ndhwc_q8(
                  convolution,
                  batchSize(),
                  inputDepth(),
                  inputHeight(),
                  inputWidth(),
                  inputPtr,
                  inputPixelStride(),
                  output.data(),
                  outputPixelStride(),
                  nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_run_operator(convolution, nullptr /* thread pool */));

          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_delete_operator(convolution));
          convolution = nullptr;
        } break;

        case Mode::Runtime:
        {
          auto packW = std::unique_ptr<qnnpack::PrePackConvWeights>(
              new qnnpack::PrePackConvWeights(
                  convolution,
                  kernelZeroPoints.data(),
                  kernel.data(),
                  bias.data()));
          ASSERT_EQ(
              pytorch_qnnp_status_success,
              qnnpack::qnnpackConv(
                  convolution,
                  packW->getPackedWeights(),
                  batchSize(),
                  inputDepth(),
                  inputHeight(),
                  inputWidth(),
                  inputZeroPoint,
                  inputPtr,
                  kernelZeroPoints.data(),
                  requantization_scales.data(),
                  outputZeroPoint,
                  qmin(),
                  qmax(),
                  output.data(),
                  nullptr));
          ASSERT_EQ(
              pytorch_qnnp_status_success,
              pytorch_qnnp_delete_operator(convolution));
        }
        break;

        default:
          // Undefined!
          ASSERT_TRUE(false);
      }

      for (size_t i = 0; i < batchSize(); i++) {
        for (size_t z = 0; z < outputDepth(); z++) {
          for (size_t y = 0; y < outputHeight(); y++) {
            for (size_t x = 0; x < outputWidth(); x++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t c = 0; c < groupOutputChannels(); c++) {
                  const double scaledAccumulator =
                      ((double)accumulators
                           [((((i * outputDepth() + z) * outputHeight() + y) *
                                  outputWidth() +
                              x) *
                                 groups() +
                             g) *
                                groupOutputChannels() +
                            c]) *
                      requantization_scales[g * groupOutputChannels() + c];
                  const double clampedAccumulator = std::max(
                      std::min(
                          scaledAccumulator,
                          double(qmax()) - double(outputZeroPoint)),
                      double(qmin()) - double(outputZeroPoint));
                  ASSERT_NEAR(
                      clampedAccumulator,
                      (int32_t(output
                                   [(((i * outputDepth() + z) * outputHeight() +
                                      y) *
                                         outputWidth() +
                                     x) *
                                        outputPixelStride() +
                                    g * groupOutputChannels() + c]) -
                       outputZeroPoint),
                      0.9)
                      << "(x, y" << (dimensionality() == 3 ? ", z" : "")
                      << ") = (" << x << ", " << y
                      << (dimensionality() == 3 ? ", " + std::to_string(z) : "")
                      << "), group = " << g << ", channel = " << c;
                }
              }
            }
          }
        }
      }
    }
  }

 private:
  uint32_t paddingDepth_{0};
  uint32_t paddingHeight_{0};
  uint32_t paddingWidth_{0};
  size_t inputDepth_{1};
  size_t inputHeight_{1};
  size_t inputWidth_{1};
  uint32_t groups_{1};
  size_t groupInputChannels_{1};
  size_t inputPixelStride_{0};
  size_t groupOutputChannels_{1};
  size_t outputPixelStride_{0};
  size_t batchSize_{1};
  uint32_t kernelDepth_{1};
  uint32_t kernelHeight_{1};
  uint32_t kernelWidth_{1};
  uint32_t dilationDepth_{1};
  uint32_t dilationHeight_{1};
  uint32_t dilationWidth_{1};
  uint32_t subsamplingDepth_{1};
  uint32_t subsamplingHeight_{1};
  uint32_t subsamplingWidth_{1};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
  bool per_channel_{false};
  size_t dimensionality_{2}; // 2 or 3
};
