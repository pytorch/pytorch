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
#include <c10/core/ScalarType.h>

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

TEST(TestQTensor, RoundingMode) {
  // We assume that quantization is defined as:
  //   qx = clamp(round(x / scale + zero_point))
  // If the zero_point is added after rounding, the result will be wrong.
  int32_t zero_point = 5;
  std::vector<float> x_values{
    -5.5, -4.5, -3.5, -2.5, -1.5, -0.5,
    0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
  std::vector<uint8_t> qx_expect{
    0, 0, 2, 2, 4, 4,
    6, 6, 8, 8, 10, 10};  // scale = 1.0

  Tensor x = from_blob(x_values.data(), x_values.size());
  Tensor qx = x.quantize_linear(/*scale=*/1.0, zero_point);

  auto qx_data = qx.data<qint8>();
  for (int idx = 0; idx < x_values.size(); ++idx) {
    ASSERT_EQ(qx_expect[idx], qx_data[idx].val_)
      << "Tie breaking during rounding element " << idx << " failed!";
  }
}

TEST(TestQTensor, Item) {
  Tensor r = at::ones({1});
  const float scale = 1;
  const int32_t zero_point = 2;
  Tensor qr = r.quantize_linear(scale, zero_point);
  ASSERT_EQ(r.item().to<float>(), qr.item().to<float>());
}

TEST(TestQTensor, EmptyQuantized) {
  float scale = 0.5;
  int zero_point = 10;
  int val = 100;
  int numel = 10;
  Tensor q = at::_empty_affine_quantized({numel}, at::device(at::kCPU).dtype(kQInt8), scale, zero_point);
  // Assigning to QTensor
  auto* q_data = q.data<qint8>();
  for (int i = 0; i < numel; ++i) {
    q_data[i].val_ = val;
  }

  // dequantize
  auto r = q.dequantize();
  auto* r_data = r.data<float>();
  for (int i = 0; i < numel; ++i) {
    ASSERT_EQ(r_data[i], (val - zero_point) * scale);
  }
}
