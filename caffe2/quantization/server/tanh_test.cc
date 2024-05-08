#include <iostream>
#include <random>
#include "tanh.h"

#include <gtest/gtest.h>
#include "caffe2/core/logging.h"

using namespace dnnlowp;
using namespace std;

TEST(Tanh, TanhUnitTest) {
  // For 8bit, we can go down to 0.0145631
  // For 9bit in 8bit out, we can go down to 0.00976522
  for (double max_abs_err = 0.02; max_abs_err <= 0.1; max_abs_err += 0.01) {
    Tanh<uint8_t> tanh_approx(max_abs_err);
    LOG(INFO) << "max_abs_err " << max_abs_err << " x_pq "
              << tanh_approx.GetPassRegionEndDequantized() << " x_sq "
              << tanh_approx.GetSaturationRegionBegin();

    const int NSAMPLES = 1 << 16;

    std::uniform_real_distribution<float> distribution(-5., 5.);
    std::default_random_engine generator;
    float sq_err_sum = 0, max_err = 0;
    for (int i = 0; i < NSAMPLES; ++i) {
      float x = distribution(generator);
      uint8_t x_q = fbgemm::Quantize<uint8_t>(
          x, tanh_approx.GetInputQuantizationParams());
      uint8_t y_q = tanh_approx.Compute(x_q);
      float y = fbgemm::Dequantize<uint8_t>(
          y_q, tanh_approx.GetOutputQuantizationParams());
      float err = fabs(tanh(x) - y);
      sq_err_sum += err * err;
      max_err = std::max(err, max_err);
      if (err > max_abs_err) {
        LOG(INFO) << "x " << x << " tanh_real " << tanh(x) << " tanh_approx "
                  << y << " err " << err << " x_q " << (int)x_q << " y_q "
                  << (int)y_q;
      }
      EXPECT_LE(err, max_abs_err);
    }
    LOG(INFO) << "avg_l2_err " << std::sqrt(sq_err_sum) / NSAMPLES
              << " max_err " << max_err << endl;
  }
}
