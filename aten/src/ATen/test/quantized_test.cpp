#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/test/test_assert.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>
// For quantize_uint8
#include <ATen/quantized/Quantizer.h>

using namespace at;

TEST(TestQTensor, QuantDequantAPIs) {
  auto num_elements = 10;
  Tensor r = at::ones({num_elements});
  const float scale = 1.0;
  const int32_t zero_point = 2;
  Tensor qr = r.quantize_linear(scale, zero_point);
  ASSERT_EQ(qr.q_scale().to<float>(), scale);
  ASSERT_EQ(qr.q_zero_point().to<int32_t>(), zero_point);
  ASSERT_TRUE(qr.is_quantized());
  ASSERT_FALSE(r.is_quantized());

  // int_repr
  Tensor int_repr = qr.int_repr();
  auto* int_repr_data = int_repr.data<uint8_t>();
  for (auto i = 0; i < num_elements; ++i) {
    ASSERT_EQ(int_repr_data[i], 3);
  }

  // Check for correct quantization
  auto r_data = r.data<float>();
  auto qr_data = qr.data<qint8>();
  for (auto i = 0; i < num_elements; ++i) {
    ASSERT_EQ(
        quantize_uint8(scale, zero_point, r_data[i]).val_, qr_data[i].val_);
  }

  // Check for correct dequantization
  Tensor rqr = qr.dequantize();
  auto rqr_data = rqr.data<float>();
  for (auto i = 0; i < num_elements; ++i) {
    ASSERT_EQ(r_data[i], rqr_data[i]);
  }
}

TEST(TestQTensor, Item) {
  Tensor r = at::ones({1});
  const float scale = 1;
  const int32_t zero_point = 2;
  Tensor qr = r.quantize_linear(scale, zero_point);
  ASSERT_EQ(r.item().to<float>(), qr.item().to<float>());
}
