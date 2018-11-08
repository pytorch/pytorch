#include "sigmoid.h"
#include <iostream>
#include <random>

#include <gtest/gtest.h>
#include "caffe2/core/logging.h"

using namespace dnnlowp;
using namespace std;

TEST(Sigmoid, SigmoidUnitTest) {
  for (double max_abs_err = 0.02; max_abs_err <= 0.1; max_abs_err += 0.01) {
    Sigmoid<uint8_t> sigmoid_approx(max_abs_err);
    LOG(INFO) << "max_abs_err " << max_abs_err;

    const int NSAMPLES = 1 << 16;

    QuantizationFactory *qfactory = QuantizationFactory::GetDefaultInstance();

    std::uniform_real_distribution<float> distribution(-5., 5.);
    std::default_random_engine generator;
    float sq_err_sum = 0, max_err = 0;
    for (int i = 0; i < NSAMPLES; ++i) {
      float x = distribution(generator);
      uint8_t x_q =
          Quantize<uint8_t>(x, sigmoid_approx.GetInputQuantizationParams());
      uint8_t y_q = sigmoid_approx.Compute(x_q);
      float y = Dequantize(y_q, sigmoid_approx.GetOutputQuantizationParams());
      float sigmoid = exp(x)/(exp(x) + 1);
      float err = fabs(sigmoid - y);
      sq_err_sum += err*err;
      max_err = std::max(err, max_err);
      if (err > max_abs_err) {
        LOG(INFO) <<
          "x " << x << " sigmoid_real " << sigmoid << " sigmoid_approx " << y <<
          " err " << err << " x_q " << (int)x_q << " y_q " << (int)y_q;
      }
      EXPECT_LE(err, max_abs_err);
    }
    LOG(INFO) << "avg_l2_err " << std::sqrt(sq_err_sum)/NSAMPLES <<
                 " max_err " << max_err;
  }
}
