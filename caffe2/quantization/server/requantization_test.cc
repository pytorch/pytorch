#include "dnnlowp.h"

#include <iostream>
#include <random>
#include <cmath>

#include <gtest/gtest.h>
#include "caffe2/core/logging.h"

using namespace std;
using namespace dnnlowp;

TEST(Requantization, BatchRequantizationUnitTest) {
  // generate input data
  default_random_engine eng;

  uniform_int_distribution<int32_t> in_max_dis(
    10, numeric_limits<int32_t>::max());
  uniform_int_distribution<int> zero_point_dis(0, 255);

  constexpr int NITER = 1024;
  constexpr int LEN = 77;

  vector<int32_t> src(LEN);
  vector<uint8_t> expected(LEN), actual(LEN);

  QuantizationFactory *qfactory = QuantizationFactory::GetDefaultInstance();

  for (int i = 0; i < NITER; ++i) {
    int32_t in_max = in_max_dis(eng);
    uniform_int_distribution<int32_t> in_dis(-in_max, in_max);

    for (int j = 0; j < LEN; ++j) {
      src[j] = in_dis(eng);
    }

    // Precise real_multiplier will be (255 / in_max) but intentionally use
    // a bigger multiplier to test if saturation is handled correctly.
    float real_multiplier = 255/(1.5 * in_max);
    TensorQuantizationParams target_qparams;
    target_qparams.zero_point = zero_point_dis(eng);
    target_qparams.precision = 8;

    RequantizationParams params = qfactory->ChooseRequantizationMultiplier(
      real_multiplier, target_qparams);

    for (int j = 0; j < LEN; ++j) {
      expected[j] = clamp(
        target_qparams.zero_point +
          std::round((double)src[j] * real_multiplier),
        8);
    }

    unsigned long long cycle_begin = __rdtsc();
    Requantize(src.data(), actual.data(), LEN, params);
    unsigned long long cycle_end = __rdtsc();
    double elements_per_cycle = (double)LEN / (cycle_end - cycle_begin);
    LOG(INFO) << elements_per_cycle << " elements_per_cycle";

    for (int j = 0; j < LEN; ++j) {
      EXPECT_EQ((int)expected[j], (int)actual[j]) <<
        "i " << i << " j " << j << " src " << src[j] <<
        " real_multiplier " << real_multiplier <<
        " multiplier " << params.multiplier <<
        " right_shift " << params.right_shift <<
        " zero_point " << target_qparams.zero_point;
    }
  }
}

TEST(Requantization, RequantizationUnitTest) {
  // Rescaling to a random range [min1, max1] to [min2, max2].
  // Make sure the ranges include 0 and inputs don't have input quantization
  // error
  default_random_engine gen;
  QuantizationFactory *qfactory = QuantizationFactory::GetDefaultInstance();

  {
    // Test 31-bit to 8-bit scaling (the most common one for example used for
    // the results of GEMM).
    // Dest quantization parameter is pre-determined by actual min/max of the
    // values.
    // Source scale can vary and zero_offset is 0.
    uniform_real_distribution<float> src_scale_exponent_dist(-19, -1);
    uniform_real_distribution<float> dst_exponent_dist(0.1, 4);
      // Bits used in src_scale plus dst should be <= 23 not to have any
      // input quantization error because float has 23 bit precision.
    uniform_real_distribution<float> negative_proportion_dist(0, 1);

    for (int i = 0; i < 256; ++i) {
      TensorQuantizationParams src_qparams;
      // scale is 2^-1 ~ 2^-19
      src_qparams.scale = powf(2, src_scale_exponent_dist(gen));
      src_qparams.zero_point = 0;
      src_qparams.precision = 31;

      float dst_extend = powf(2, dst_exponent_dist(gen));
      float negative_proportion = negative_proportion_dist(gen);
      float min = -(dst_extend * negative_proportion);
      float max = dst_extend + min;
      TensorQuantizationParams dst_qparams =
        qfactory->ChooseQuantizationParams(min, max);
        // scale = dst_extend / 2^8
        // which is between 0.1/2^-8 ~ 2^-4

      float real_multiplier = src_qparams.scale / dst_qparams.scale;
      RequantizationParams requantization_params =
        qfactory->ChooseRequantizationMultiplier(real_multiplier, dst_qparams);

      uniform_real_distribution<float> value_dist(
          ceil(min/src_qparams.scale)*src_qparams.scale,
          floor(max/src_qparams.scale)*src_qparams.scale);
        // round with src_qparams.scale to avoid input quantization error due
        // to clipping
      float sum_sq = 0, max_err = 0;
      constexpr int LEN = 1111;
      vector<int32_t> src_q(LEN);
      vector<float> src(LEN);
      for (int j = 0; j < LEN; ++j) {
        float src_orig = value_dist(gen);
        src_q[j] = Quantize<int32_t>(
          src_orig, 0, src_qparams.scale, 32, true /* signed*/);
        src[j] = Dequantize<int32_t>(src_q[j], src_qparams);
          // This number shouldn't have any quantization error
        EXPECT_EQ(
          Quantize<int32_t>(src[j], 0, src_qparams.scale, 32, true), src_q[j]);
      }

      vector<uint8_t> dst_q(LEN);
      Requantize(src_q.data(), dst_q.data(), LEN, requantization_params);

      for (int j = 0; j < LEN; ++j) {
        float dst = Dequantize<uint8_t>(dst_q[j], dst_qparams);

        float err = fabsf(dst - src[j]);
        sum_sq += err * err;
        max_err = std::max(max_err, err);
        EXPECT_LE(err, dst_qparams.scale / 1.9);
      }

      LOG(INFO) <<
        "src_scale " << src_qparams.scale << " dst_extend " << dst_extend <<
        " real_multiplier " << real_multiplier <<
        " avg_l2_err " << std::sqrt(sum_sq) / 1024 << " max_err " << max_err <<
        endl;
      // We shouldn't have an error bigger than output quantization error
      EXPECT_LE(max_err, dst_qparams.scale / 1.9);
    }
  }
}
