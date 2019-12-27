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
#include <qnnpack/pack.h>
#include <qnnpack/params.h>
#include <qnnpack/requantization.h>

class DWConvMicrokernelTester {
 public:
  inline DWConvMicrokernelTester& width(uint32_t width) {
    assert(width >= 1);
    this->width_ = width;
    return *this;
  }

  inline uint32_t width() const {
    return this->width_;
  }

  inline DWConvMicrokernelTester& subsampling(uint32_t subsampling) {
    assert(subsampling >= 1);
    this->subsampling_ = subsampling;
    return *this;
  }

  inline uint32_t subsampling() const {
    return this->subsampling_;
  }

  inline DWConvMicrokernelTester& channels(uint32_t channels) {
    assert(channels >= 1);
    this->channels_ = channels;
    return *this;
  }

  inline uint32_t channels() const {
    return this->channels_;
  }

  inline DWConvMicrokernelTester& cr(uint32_t cr) {
    assert(cr != 0);
    assert((cr & (cr - 1)) == 0);
    this->cr_ = cr;
    return *this;
  }

  inline uint32_t cr() const {
    return this->cr_;
  }

  inline uint32_t packedChannels() const {
    return (channels() + (cr() - 1)) & -cr();
  }

  inline DWConvMicrokernelTester& kernelHeight(uint32_t kernelHeight) {
    assert(kernelHeight != 0);
    this->kernelHeight_ = kernelHeight;
    return *this;
  }

  inline uint32_t kernelHeight() const {
    return this->kernelHeight_;
  }

  inline DWConvMicrokernelTester& kernelWidth(uint32_t kernelWidth) {
    assert(kernelWidth != 0);
    this->kernelWidth_ = kernelWidth;
    return *this;
  }

  inline uint32_t kernelWidth() const {
    return this->kernelWidth_;
  }

  inline uint32_t kernelSize() const {
    return kernelHeight() * kernelWidth();
  }

  inline DWConvMicrokernelTester& inputStride(uint32_t inputStride) {
    assert(inputStride != 0);
    this->inputStride_ = inputStride;
    return *this;
  }

  inline uint32_t inputStride() const {
    if (this->inputStride_ == 0) {
      return channels();
    } else {
      assert(this->inputStride_ >= channels());
      return this->inputStride_;
    }
  }

  inline DWConvMicrokernelTester& outputStride(uint32_t outputStride) {
    assert(outputStride != 0);
    this->outputStride_ = outputStride;
    return *this;
  }

  inline uint32_t outputStride() const {
    if (this->outputStride_ == 0) {
      return channels();
    } else {
      assert(this->outputStride_ >= channels());
      return this->outputStride_;
    }
  }

  inline DWConvMicrokernelTester& inputZeroPoint(uint8_t inputZeroPoint) {
    this->inputZeroPoint_ = inputZeroPoint;
    return *this;
  }

  inline uint8_t inputZeroPoint() const {
    return this->inputZeroPoint_;
  }

  inline DWConvMicrokernelTester& kernelZeroPoint(uint8_t kernelZeroPoint) {
    this->kernelZeroPoint_ = kernelZeroPoint;
    return *this;
  }

  inline uint8_t kernelZeroPoint() const {
    return this->kernelZeroPoint_;
  }

  inline DWConvMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline DWConvMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline DWConvMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void test(pytorch_q8dwconv_up_ukernel_function q8dwconv) const {
    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> input(
        (kernelSize() + (width() * subsampling() - 1) * kernelHeight() - 1) *
            inputStride() +
        channels() + 8);
    std::vector<uint8_t> kernel(channels() * kernelSize());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedWeights(
        (kernelSize() + sizeof(int32_t) / sizeof(uint8_t)) * packedChannels());
    std::vector<int32_t> bias(packedChannels());
    std::vector<int32_t> accumulators(width() * channels());
    std::vector<uint8_t> output((width() - 1) * outputStride() + channels());
    std::vector<const uint8_t*> indirectInput(
        kernelSize() + (width() * subsampling() - 1) * kernelHeight());

    const uint8_t* inputPtr = input.data() + 8;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(accumulators.begin(), accumulators.end(), 0);

      ASSERT_NE(
          *std::max_element(input.cbegin(), input.cend()),
          *std::min_element(input.cbegin(), input.cend()));
      ASSERT_NE(
          *std::max_element(kernel.cbegin(), kernel.cend()),
          *std::min_element(kernel.cbegin(), kernel.cend()));

      std::fill(packedWeights.begin(), packedWeights.end(), 0xA5);

      pytorch_pack_q8dw_w(
          kernelHeight(),
          kernelWidth(),
          channels(),
          cr(),
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
          inputZeroPoint(),
          kernelZeroPoint(),
#endif
          kernel.data(),
          bias.data(),
          packedWeights.data());

      for (size_t i = 0;
           i < kernelSize() + (width() * subsampling() - 1) * kernelHeight();
           i++) {
        indirectInput[i] = inputPtr + i * inputStride();
      }
      std::shuffle(indirectInput.begin(), indirectInput.end(), rng);

      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          int32_t acc = bias[c];
          for (size_t kx = 0; kx < kernelWidth(); kx++) {
            for (size_t ky = 0; ky < kernelHeight(); ky++) {
              acc += (int32_t(indirectInput
                                  [(x * subsampling() + kx) * kernelHeight() +
                                   ky][c]) -
                      int32_t(inputZeroPoint())) *
                  (int32_t(
                       kernel[(c * kernelHeight() + ky) * kernelWidth() + kx]) -
                   int32_t(kernelZeroPoint()));
            }
          }
          accumulators[x * channels() + c] = acc;
        }
      }
      const int32_t accumulatorsMin =
          *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulatorsMax =
          *std::max_element(accumulators.cbegin(), accumulators.cend());
      const uint32_t accumulatorsRange =
          uint32_t(accumulatorsMax) - uint32_t(accumulatorsMin);
      ASSERT_NE(0, accumulatorsRange);

      const double outputScale = accumulatorsRange >= 256
          ? double(accumulatorsRange) / 255.0
          : 1.00001;
      const uint8_t outputZeroPoint = uint8_t(std::max(
          std::min(
              lrint(
                  127.5 -
                  0.5 * double(accumulatorsMin + accumulatorsMax) /
                      outputScale),
              long(std::numeric_limits<uint8_t>::max())),
          long(std::numeric_limits<uint8_t>::min())));

      const float requantizationScale = 1.0f / float(outputScale);
      const union pytorch_qnnp_conv_quantization_params quantizationParams =
          pytorch_qnnp_compute_conv_quantization_params(
              inputZeroPoint(),
              kernelZeroPoint(),
              requantizationScale,
              outputZeroPoint,
              qmin(),
              qmax());
      const union pytorch_qnnp_q31_requantization_params
          scalarRequantizationParams =
              pytorch_qnnp_compute_scalar_requantization_params(
                  requantizationScale, outputZeroPoint, qmin(), qmax());

      q8dwconv(
          channels(),
          width(),
          indirectInput.data(),
          packedWeights.data(),
          output.data(),
          kernelHeight() * subsampling() * sizeof(void*),
          (outputStride() - channels()) * sizeof(uint8_t),
          &quantizationParams);

      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          const uint8_t referenceOutput = pytorch_qnnp_q31_requantize(
              accumulators[x * channels() + c], scalarRequantizationParams);
          const double scaledAccumulator =
              accumulators[x * channels() + c] / outputScale +
              double(outputZeroPoint);
          const double clampedAccumulator = std::max(
              std::min(scaledAccumulator, double(qmax())), double(qmin()));
          ASSERT_NEAR(
              clampedAccumulator, double(output[x * outputStride() + c]), 0.6)
              << "x = " << x << ", channel = " << c;
          ASSERT_EQ(
              uint32_t(referenceOutput),
              uint32_t(output[x * outputStride() + c]))
              << "x = " << x << ", channel = " << c;
        }
      }
    }
  }

  void test(pytorch_q8dwconv_mp_ukernel_function q8dwconv) const {
    ASSERT_EQ(25, kernelSize())
        << "only 5x5 microkernel is currently supported";

    std::random_device randomDevice;
    auto rng = std::mt19937(randomDevice());
    auto s32rng =
        std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint8_t>(), rng);

    std::vector<uint8_t> input(
        (kernelSize() + (width() * subsampling() - 1) * kernelHeight() - 1) *
            inputStride() +
        channels() + 8);
    std::vector<uint8_t> kernel(channels() * kernelSize());
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> packedWeights(
        (kernelSize() + sizeof(int32_t) / sizeof(uint8_t)) * packedChannels());
    std::vector<int32_t> bias(packedChannels());
    std::vector<int32_t> accumulators(width() * channels());
    std::vector<int32_t> mpAcc(width() * packedChannels());
    std::vector<uint8_t> output((width() - 1) * outputStride() + channels());
    std::vector<const uint8_t*> indirectInput(
        kernelSize() + (width() * subsampling() - 1) * kernelHeight());

    const uint8_t* inputPtr = input.data() + 8;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(accumulators.begin(), accumulators.end(), 0);
      std::fill(mpAcc.begin(), mpAcc.end(), 0xA5A55A5A);

      ASSERT_NE(
          *std::max_element(input.cbegin(), input.cend()),
          *std::min_element(input.cbegin(), input.cend()));
      ASSERT_NE(
          *std::max_element(kernel.cbegin(), kernel.cend()),
          *std::min_element(kernel.cbegin(), kernel.cend()));

      std::fill(packedWeights.begin(), packedWeights.end(), 0xA5);

      ASSERT_EQ(25, kernelSize())
          << "only 5x5 microkernel is currently supported";
      pytorch_pack_q8dw_w_dilation(
          kernelHeight(),
          kernelWidth(),
          channels(),
          cr(),
          0,
          kernelHeight(),
          0,
          2,
          kernel.data(),
          bias.data(),
          packedWeights.data(),
          true);
      pytorch_pack_q8dw_w_dilation(
          kernelHeight(),
          kernelWidth(),
          channels(),
          cr(),
          0,
          kernelHeight(),
          2,
          4,
          kernel.data(),
          bias.data(),
          packedWeights.data() +
              (10 + sizeof(int32_t) / sizeof(uint8_t)) * packedChannels(),
          false);
      pytorch_pack_q8dw_w_dilation(
          kernelHeight(),
          kernelWidth(),
          channels(),
          cr(),
          0,
          kernelHeight(),
          4,
          5,
          kernel.data(),
          bias.data(),
          packedWeights.data() +
              (20 + sizeof(int32_t) / sizeof(uint8_t)) * packedChannels(),
          false);
      for (size_t i = 0;
           i < kernelSize() + (width() * subsampling() - 1) * kernelHeight();
           i++) {
        indirectInput[i] = inputPtr + i * inputStride();
      }
      std::shuffle(indirectInput.begin(), indirectInput.end(), rng);

      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          int32_t acc = bias[c];
          for (size_t kx = 0; kx < kernelWidth(); kx++) {
            for (size_t ky = 0; ky < kernelHeight(); ky++) {
              acc += (int32_t(indirectInput
                                  [(x * subsampling() + kx) * kernelHeight() +
                                   ky][c]) -
                      int32_t(inputZeroPoint())) *
                  (int32_t(
                       kernel[(c * kernelHeight() + ky) * kernelWidth() + kx]) -
                   int32_t(kernelZeroPoint()));
            }
          }
          accumulators[x * channels() + c] = acc;
        }
      }
      const int32_t accumulatorsMin =
          *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulatorsMax =
          *std::max_element(accumulators.cbegin(), accumulators.cend());
      const uint32_t accumulatorsRange =
          uint32_t(accumulatorsMax) - uint32_t(accumulatorsMin);
      ASSERT_NE(0, accumulatorsRange);

      const double outputScale = accumulatorsRange >= 256
          ? double(accumulatorsRange) / 255.0
          : 1.00001;
      const uint8_t outputZeroPoint = uint8_t(std::max(
          std::min(
              lrint(
                  127.5 -
                  0.5 * double(accumulatorsMin + accumulatorsMax) /
                      outputScale),
              long(std::numeric_limits<uint8_t>::max())),
          long(std::numeric_limits<uint8_t>::min())));

      const float requantizationScale = 1.0f / float(outputScale);
      const union pytorch_qnnp_conv_quantization_params quantizationParams =
          pytorch_qnnp_compute_conv_quantization_params(
              inputZeroPoint(),
              kernelZeroPoint(),
              requantizationScale,
              outputZeroPoint,
              qmin(),
              qmax());
      const union pytorch_qnnp_q31_requantization_params
          scalarRequantizationParams =
              pytorch_qnnp_compute_scalar_requantization_params(
                  requantizationScale, outputZeroPoint, qmin(), qmax());

      q8dwconv(
          channels(),
          width(),
          indirectInput.data(),
          packedWeights.data(),
          mpAcc.data(),
          output.data(),
          kernelHeight() * subsampling() * sizeof(void*),
          (outputStride() - channels()) * sizeof(uint8_t),
          &quantizationParams);

      for (size_t x = 0; x < width(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          const uint8_t referenceOutput = pytorch_qnnp_q31_requantize(
              accumulators[x * channels() + c], scalarRequantizationParams);
          const double scaledAccumulator =
              accumulators[x * channels() + c] / outputScale +
              double(outputZeroPoint);
          const double clampedAccumulator = std::max(
              std::min(scaledAccumulator, double(qmax())), double(qmin()));
          ASSERT_NEAR(
              clampedAccumulator, double(output[x * outputStride() + c]), 0.6)
              << "x = " << x << ", channel = " << c;
          ASSERT_EQ(
              uint32_t(referenceOutput),
              uint32_t(output[x * outputStride() + c]))
              << "x = " << x << ", channel = " << c;
        }
      }
    }
  }

 private:
  uint32_t channels_{1};
  uint32_t cr_{1};
  uint32_t width_{1};
  uint32_t subsampling_{1};
  uint32_t kernelHeight_{1};
  uint32_t kernelWidth_{1};
  uint32_t inputStride_{0};
  uint32_t outputStride_{0};
  uint8_t inputZeroPoint_{127};
  uint8_t kernelZeroPoint_{127};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{3};
};
