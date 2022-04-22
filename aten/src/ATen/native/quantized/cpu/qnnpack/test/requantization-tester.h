/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdlib>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <qnnpack/params.h>
#include <qnnpack/scalar-utils.h>

class RequantizationTester {
 public:
  inline RequantizationTester& s(uint32_t s) {
    this->s_ = s;
    return *this;
  }

  inline uint32_t s() const {
    return this->s_;
  }

  inline float scale() const {
    return ldexpf(1.0f, -s());
  }

  inline RequantizationTester& zeroPoint(int32_t zeroPoint) {
    this->zeroPoint_ = zeroPoint;
    return *this;
  }

  inline int32_t zeroPoint() const {
    return this->zeroPoint_;
  }

  inline RequantizationTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline RequantizationTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline RequantizationTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  /*
   * Test that requantization of numbers ((i - zero point) * 2**s) with
   * - scale = exp2(-s)
   * - zero point in [0, 255]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not
   * overflow.
   */
  void testExactDivideByPO2(pytorch_requantization_function requantize) const {
    ASSERT_GE(zeroPoint(), 0);
    ASSERT_LE(zeroPoint(), 255);

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());
    const int32_t maxI =
        (uint32_t(std::numeric_limits<int32_t>::max()) >> s()) + zeroPoint();
    const int32_t minI =
        -(-uint32_t(std::numeric_limits<int32_t>::min()) >> s()) + zeroPoint();
    for (int32_t i = 0; i < 256; i++) {
      const int32_t clampedI = std::max(minI, std::min(maxI, i));
      inputs[i] = int32_t(uint32_t(clampedI - zeroPoint()) << s());
    }
    requantize(
        inputs.size(),
        inputs.data(),
        scale(),
        zeroPoint(),
        qmin(),
        qmax(),
        outputs.data());
    for (int32_t i = 0; i < 256; i++) {
      const int32_t clampedI = std::max(minI, std::min(maxI, i));
      ASSERT_EQ(clampedI, outputs[i])
          << "i = " << i << ", clamped i = " << clampedI << ", min i = " << minI
          << ", max i = " << maxI << ", s = " << s()
          << ", zero point = " << zeroPoint();
    }
  }

  /*
   * Test that requantization of numbers (i * 2**s + sign(i - zero point) *
   * 2**(s-1)) with
   * - scale = exp2(-s)
   * - zero point in [1, 255]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not
   * overflow.
   */
  void testDivideByPO2WithRoundingUp(pytorch_requantization_function requantize) {
    ASSERT_GE(zeroPoint(), 0);
    ASSERT_LE(zeroPoint(), 255);

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());
    for (int32_t i = 0; i < 256; i++) {
      const int64_t input =
          RequantizationTester::shiftLeft(i - zeroPoint(), s()) -
          (INT64_C(1) << (s() - 1)) + (int64_t)(i <= zeroPoint());
      inputs[i] = int32_t(input);
    }
    requantize(
        inputs.size(),
        inputs.data(),
        scale(),
        zeroPoint(),
        qmin(),
        qmax(),
        outputs.data());
    for (int32_t i = 0; i < 256; i++) {
      const int64_t input =
          RequantizationTester::shiftLeft(i - zeroPoint(), s()) -
          (INT64_C(1) << (s() - 1)) + (int64_t)(i <= zeroPoint());
      if (int32_t(input) == input) {
        ASSERT_EQ(i, uint32_t(outputs[i]))
            << "i = " << i << ", input = " << input << ", s = " << s()
            << ", zero point = " << zeroPoint();
      }
    }
  }

  /*
   * Test that requantization of numbers (i * 2**s + sign(i - zero point) *
   * 2**(s-1)) with
   * - scale = exp2(-s)
   * - zero point in [1, 255]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not
   * overflow.
   */
  void testDivideByPO2WithRoundingDown(pytorch_requantization_function requantize) {
    ASSERT_GE(zeroPoint(), 0);
    ASSERT_LE(zeroPoint(), 255);

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());
    for (int32_t i = 0; i < 256; i++) {
      const int64_t input =
          RequantizationTester::shiftLeft(i - zeroPoint(), s()) +
          (INT64_C(1) << (s() - 1)) - (int64_t)(i >= zeroPoint());
      inputs[i] = int32_t(input);
    }
    requantize(
        inputs.size(),
        inputs.data(),
        scale(),
        zeroPoint(),
        qmin(),
        qmax(),
        outputs.data());
    for (int32_t i = 0; i < 256; i++) {
      const int64_t input =
          RequantizationTester::shiftLeft(i - zeroPoint(), s()) +
          (INT64_C(1) << (s() - 1)) - (int64_t)(i >= zeroPoint());
      if (int32_t(input) == input) {
        ASSERT_EQ(i, uint32_t(outputs[i]))
            << "i = " << i << ", input = " << input << ", s = " << s()
            << ", zero point = " << zeroPoint();
      }
    }
  }

  void testDivideByPO2WithRoundingAway(pytorch_requantization_function requantize) {
    ASSERT_GE(zeroPoint(), 0);
    ASSERT_LE(zeroPoint(), 255);

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());
    for (int32_t i = 0; i < 256; i++) {
      int64_t input = RequantizationTester::shiftLeft(i - zeroPoint(), s());
      if (input > 0) {
        input -= INT64_C(1) << (s() - 1);
      } else if (input < 0) {
        input += INT64_C(1) << (s() - 1);
      }
      inputs[i] = int32_t(input);
    }
    requantize(
        inputs.size(),
        inputs.data(),
        scale(),
        zeroPoint(),
        qmin(),
        qmax(),
        outputs.data());
    for (uint32_t i = 0; i < 256; i++) {
      int64_t input = RequantizationTester::shiftLeft(i - zeroPoint(), s());
      if (input > 0) {
        input -= INT64_C(1) << (s() - 1);
      } else if (input < 0) {
        input += INT64_C(1) << (s() - 1);
      }
      if (int32_t(input) == input) {
        ASSERT_EQ(i, uint32_t(outputs[i]))
            << "i = " << i << ", input = " << input << ", s = " << s()
            << ", zero point = " << zeroPoint();
      }
    }
  }

  void testSpecialCases(pytorch_requantization_function requantize) {
    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());

    std::fill(
        inputs.begin(), inputs.end(), std::numeric_limits<int32_t>::min());
    for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
      requantize(
          inputs.size(),
          inputs.data(),
          ldexpf(1.0f, -32) /* scale */,
          zeroPoint /* zero point */,
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(),
          outputs.data());
      ASSERT_EQ(
          std::max(int32_t(0), zeroPoint - 1),
          *std::min_element(outputs.cbegin(), outputs.cend()));
    }

    std::fill(
        inputs.begin(), inputs.end(), std::numeric_limits<int32_t>::max());
    requantize(
        inputs.size(),
        inputs.data(),
        0x1.FFFFFEp-1f /* scale */,
        std::numeric_limits<uint8_t>::max() /* zero point */,
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max(),
        outputs.data());
    for (size_t i = 0; i < inputs.size(); i++) {
      ASSERT_EQ(std::numeric_limits<uint8_t>::max(), outputs[i]);
    }
  }

  void testRandomCasesPrecise(pytorch_requantization_function requantize) {
    std::random_device randomDevice;
    std::mt19937 mtRng(randomDevice());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      auto rng = std::bind(std::uniform_int_distribution<uint8_t>(), mtRng);

      std::vector<int32_t> inputs(4096);
      std::vector<uint8_t> outputs(inputs.size());

      const uint8_t zeroPoint = UINT8_C(128);
      std::uniform_real_distribution<float> scaleDistribution(
          0x1.000000p-23f, 0x1.FFFFFEp-1f);
      const float scale = scaleDistribution(mtRng);
      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t approximateOutput = rng();
        const int32_t input =
            int32_t(double(approximateOutput) / double(scale));
        inputs[i] = input;
      }

      requantize(
          inputs.size(),
          inputs.data(),
          scale,
          zeroPoint,
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(),
          outputs.data());

      /* Ensure that outputs are not all identical, as in this case test doesn't
       * validate much */
      ASSERT_NE(
          *std::max_element(outputs.cbegin(), outputs.cend()),
          *std::min_element(outputs.cbegin(), outputs.cend()));

      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t referenceOutput = pytorch_scalar_requantize_precise(
            inputs[i],
            scale,
            zeroPoint,
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max());
        ASSERT_EQ(uint32_t(referenceOutput), uint32_t(outputs[i]));
      }
    }
  }

  void testRandomCasesApproximate(pytorch_requantization_function requantize) {
    std::random_device randomDevice;
    std::mt19937 mtRng(randomDevice());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      auto rng = std::bind(std::uniform_int_distribution<uint8_t>(), mtRng);

      std::vector<int32_t> inputs(4096);
      std::vector<uint8_t> outputs(inputs.size());

      const uint8_t zeroPoint = UINT8_C(128);
      std::uniform_real_distribution<float> scaleDistribution(
          0x1.000000p-23f, 0x1.FFFFFEp-1f);
      const float scale = scaleDistribution(mtRng);
      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t approximateOutput = rng();
        const int32_t input =
            int32_t(double(approximateOutput) / double(scale));
        inputs[i] = input;
      }

      requantize(
          inputs.size(),
          inputs.data(),
          scale,
          zeroPoint,
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(),
          outputs.data());

      /* Ensure that outputs are not all identical, as in this case test doesn't
       * validate much */
      ASSERT_NE(
          *std::max_element(outputs.cbegin(), outputs.cend()),
          *std::min_element(outputs.cbegin(), outputs.cend()));

      for (size_t i = 0; i < inputs.size(); i++) {
        const double referenceOutput =
            RequantizationTester::requantizeApproximate(
                inputs[i],
                scale,
                zeroPoint,
                std::numeric_limits<uint8_t>::min(),
                std::numeric_limits<uint8_t>::max());
        ASSERT_LE(fabs(referenceOutput - double(outputs[i])), 0.55)
            << "input = " << inputs[i] << ", output = " << uint32_t(outputs[i])
            << ", reference output = " << referenceOutput;
      }
    }
  }

  void testRandomCasesAgainstReference(
      pytorch_requantization_function requantize,
      pytorch_requantization_function requantizeReference) {
    std::random_device randomDevice;
    std::mt19937 mtRng(randomDevice());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      auto rng = std::bind(std::uniform_int_distribution<uint8_t>(), mtRng);

      std::vector<int32_t> inputs(4096);
      std::vector<uint8_t> outputs(inputs.size());
      std::vector<uint8_t> referenceOutputs(inputs.size());

      const uint8_t zeroPoint = UINT8_C(128);
      std::uniform_real_distribution<float> scaleDistribution(
          0x1.000000p-23f, 0x1.FFFFFEp-1f);
      const float scale = scaleDistribution(mtRng);
      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t approximateOutput = rng();
        const int32_t input =
            int32_t(double(approximateOutput) / double(scale));
        inputs[i] = input;
      }

      requantize(
          inputs.size(),
          inputs.data(),
          scale,
          zeroPoint,
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(),
          outputs.data());

      requantizeReference(
          inputs.size(),
          inputs.data(),
          scale,
          zeroPoint,
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(),
          referenceOutputs.data());

      /* Ensure that outputs are not all identical, as in this case test doesn't
       * validate much */
      ASSERT_NE(
          *std::max_element(outputs.cbegin(), outputs.cend()),
          *std::min_element(outputs.cbegin(), outputs.cend()));

      for (size_t i = 0; i < inputs.size(); i++) {
        ASSERT_EQ(uint32_t(referenceOutputs[i]), uint32_t(outputs[i]));
      }
    }
  }

  static inline int64_t shiftLeft(int64_t w, uint32_t n) {
    return (int64_t)((uint64_t)w << n);
  }

  static inline double requantizeApproximate(
      int32_t value,
      float scale,
      uint8_t zeroPoint,
      uint8_t qmin,
      uint8_t qmax) {
    assert(scale < 1.0f);
    assert(scale >= 0x1.0p-32f);

    double clampedValue = double(value) * double(scale) + double(zeroPoint);

    const double fmin = double(qmin);
    if (clampedValue < fmin) {
      clampedValue = fmin;
    }

    const double fmax = double(qmax);
    if (clampedValue > fmax) {
      clampedValue = fmax;
    }

    return clampedValue;
  }

 private:
  size_t zeroPoint_{0};
  size_t s_{1};
  uint8_t qmin_{std::numeric_limits<uint8_t>::min()};
  uint8_t qmax_{std::numeric_limits<uint8_t>::max()};
  size_t iterations_{1};
};
