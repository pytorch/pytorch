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
  auto num_elements = 10;
  Tensor r = at::ones({num_elements});
  const float scale = 1.0;
  const int32_t zero_point = 2;
  Tensor qr = r.quantize_linear(scale, zero_point);
  ASSERT_EQ(qr.q_scale().to<float>(), scale);
  ASSERT_EQ(qr.q_zero_point().to<int32_t>(), zero_point);

  // TODO: Uncomment when quantizer is ready
  // auto* quantizer = static_cast<PerLayerAffineQuantizer*>(qr.quantizer());
  // ASSERT_EQ(quantizer->scale(), scale);
  // ASSERT_EQ(quantizer->zero_point(), zero_point);

  // Check for correct quantization
  auto r_data = r.data<float>();
  auto qr_data = qr.data<qint8>();
  for (auto i = 0; i < num_elements; ++i) {
    ASSERT_EQ(QuantizeUint8(scale, zero_point, r_data[i]), qr_data[i]);
  }

  // Check for correct dequantization
  Tensor rqr = qr.dequantize();
  auto rqr_data = rqr.data<float>();
  for (auto i = 0; i < num_elements; ++i) {
    ASSERT_EQ(r_data[i], rqr_data[i]);
  }
}
