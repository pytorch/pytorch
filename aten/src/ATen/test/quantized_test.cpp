#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <cmath>
#include <type_traits>
#include <ATen/test/test_assert.h>
#include <ATen/Quantizer.h>

using namespace at;

TEST(TestQTensor, First) {
  Tensor r = at::ones({10});
  const float scale = 1.0;
  const int32_t zero_point = 2;
  Tensor qr = r.quantize_linear(scale, zero_point);
  ASSERT_EQ(qr.q_scale().to<float>(), scale);
  ASSERT_EQ(qr.q_zero_point().to<int32_t>(), zero_point);

  auto* quantizer = static_cast<PerLayerAffineQuantizer*>(qr.quantizer());
  ASSERT_EQ(quantizer->scale(), scale);
  ASSERT_EQ(quantizer->zero_point(), zero_point);

  Tensor rqr = qr.dequantize();
  auto rqr_a = rqr.data<float>();
  auto r_a = r.data<float>();
  for (auto i = 0; i < 10; ++i) {
    ASSERT_EQ(r_a[i], rqr_a[i]);
  }
}
