#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/test/test_assert.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>
// For quantize_val
#include <ATen/native/quantized/affine_quantizer.h>
#include <c10/core/ScalarType.h>

using namespace at;

TEST(TestQTensor, QuantDequantAPIs) {
  auto num_elements = 10;
  Tensor r = at::ones({num_elements});
  const double scale = 1.0;
  const int64_t zero_point = 2;
  const Tensor qr = at::quantize_per_tensor(r, scale, zero_point, kQUInt8);
  ASSERT_EQ(qr.q_scale(), scale);
  ASSERT_EQ(qr.q_zero_point(), zero_point);
  ASSERT_TRUE(qr.is_quantized());
  ASSERT_FALSE(r.is_quantized());

  // int_repr
  Tensor int_repr = qr.int_repr();
  auto* int_repr_data = int_repr.data_ptr<uint8_t>();
  for (auto i = 0; i < num_elements; ++i) {
    ASSERT_EQ(int_repr_data[i], 3);
  }

  // Check for correct quantization
  auto r_data = r.data_ptr<float>();
  auto qr_data = qr.data_ptr<quint8>();
  for (auto i = 0; i < num_elements; ++i) {
    ASSERT_EQ(
        native::quantize_val<quint8>(scale, zero_point, r_data[i]).val_,
        qr_data[i].val_);
  }

  // Check for correct dequantization
  Tensor rqr = qr.dequantize();
  auto rqr_data = rqr.data_ptr<float>();
  for (auto i = 0; i < num_elements; ++i) {
    ASSERT_EQ(r_data[i], rqr_data[i]);
  }
  for (auto i = 0; i < num_elements; ++i) {
    ASSERT_EQ(
        r_data[i],
        native::dequantize_val(qr.q_scale(), qr.q_zero_point(), qr_data[i]));
  }

  // Check for correct requantization
  double new_scale = 2.0;
  int64_t new_zero_point = 1;
  Tensor reqr = at::quantize_per_tensor(r, new_scale, new_zero_point, kQInt8);
  auto reqr_data = reqr.data_ptr<qint8>();
  for (auto i = 0; i < num_elements; ++i) {
    reqr_data[i].val_ =
        native::requantize_val<quint8, qint8>(
            scale, zero_point, new_scale, new_zero_point, qr_data[i])
            .val_;
    const qint8 expected =
        native::quantize_val<qint8>(new_scale, new_zero_point, rqr_data[i]);
    ASSERT_EQ(expected.val_, reqr_data[i].val_);
  }
}

TEST(TestQTensor, RoundingMode) {
  // We assume that quantization is defined as:
  //   qx = clamp(zero_point + round(x / scale))
  // If the zero_point is added before rounding, the result will be wrong.
  int32_t zero_point = 5;
  std::vector<float> x_values{
      -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
  std::vector<uint8_t> qx_expect{
      0, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11}; // scale = 1.0

  Tensor x = from_blob(x_values.data(), x_values.size());
  Tensor qx = at::quantize_per_tensor(x, /*scale=*/1.0, zero_point, kQUInt8);

  auto qx_data = qx.data_ptr<quint8>();
  for (size_t idx = 0; idx < x_values.size(); ++idx) {
    ASSERT_EQ(qx_expect[idx], qx_data[idx].val_)
        << "Tie breaking during rounding element " << idx << " failed!";
  }
}

TEST(TestQTensor, Item) {
  Tensor r = at::ones({1});
  const float scale = 1;
  const int32_t zero_point = 2;
  Tensor qr = at::quantize_per_tensor(r, scale, zero_point, kQUInt8);
  ASSERT_EQ(r.item().to<float>(), qr.item().to<float>());
}

TEST(TestQTensor, EmptyQuantized) {
  float scale = 0.5;
  int zero_point = 10;
  int val = 100;
  int numel = 10;
  Tensor q = at::_empty_affine_quantized(
      {numel}, at::device(at::kCPU).dtype(kQUInt8), scale, zero_point);
  // Assigning to QTensor
  auto* q_data = q.data_ptr<quint8>();
  for (int i = 0; i < numel; ++i) {
    q_data[i].val_ = val;
  }

  // dequantize
  auto r = q.dequantize();
  auto* r_data = r.data_ptr<float>();
  for (int i = 0; i < numel; ++i) {
    ASSERT_EQ(r_data[i], (val - zero_point) * scale);
  }
}

TEST(TestQTensor, EmptyPerchannelQuantized) {
  int numel = 10;
  auto scales = rand({numel}).toType(kDouble);
  auto zero_points = randint(10, {10}).toType(kLong);
  int val = 100;
  int ch_axis = 0;
  Tensor q = at::_empty_per_channel_affine_quantized(
      {numel},
      scales,
      zero_points,
      ch_axis,
      at::device(at::kCPU).dtype(kQUInt8));
  // Assigning to QTensor
  auto* q_data = q.data_ptr<quint8>();
  for (int i = 0; i < numel; ++i) {
    q_data[i].val_ = val;
  }

  // dequantize
  auto r = q.dequantize();
  auto* r_data = r.data_ptr<float>();
  for (int i = 0; i < numel; ++i) {
    ASSERT_EQ(
        r_data[i],
        (val - zero_points[i].item().to<int>()) * scales[i].item().to<float>());
  }
}

TEST(TestQTensor, QuantizePerChannel4d) {
  int C = 64, H = 10, W = 10;
  auto scales = rand({C}).toType(kDouble);
  auto zero_points = randint(10, {C}).toType(kLong);
  int ch_axis = 1;
  // create 4d tensor where each H x W image is a range(0, H*W)
  Tensor tensor = at::empty({1, C, H, W}, at::device(at::kCPU).dtype(kFloat));
  auto* tensor_data = tensor.data_ptr<float>();
  for (int c = 0, i = 0; c < C; ++c) {
    for (int e = 0; e < H * W; ++e, ++i) {
      tensor_data[i] = e;
    }
  }
  // quantize and check values
  Tensor q = at::native::quantize_per_channel_cpu(
      tensor, scales, zero_points, ch_axis, kQUInt8);
  auto* q_data = (uint8_t*)q.data_ptr<quint8>();
  for (int c = 0, i = 0; c < C; ++c) {
    float inv_scale = 1.0f / static_cast<float>(scales[c].item<double>());
    int64_t zero_point = zero_points[c].item<int64_t>();
    for (int e = 0; e < H * W; ++e, ++i) {
      // downsize qval to 255 if val is greater than max uint8_t value
      int qval = std::min<int>(zero_point + std::nearbyint(e * inv_scale), 255);
      ASSERT_EQ((int)q_data[i], qval);
    }
  }
}

TEST(TestQTensor, QuantizePerChannel4dChannelsLast) {
  int C = 64, H = 10, W = 10;
  auto scales = rand({C}).toType(kDouble);
  auto zero_points = randint(10, {C}).toType(kLong);
  int ch_axis = 1;
  // create 4d tensor where each H x W image is a range(0, H*W)
  Tensor tensor = at::empty(
      {1, C, H, W},
      at::device(at::kCPU).dtype(kFloat).memory_format(
          at::MemoryFormat::ChannelsLast));
  auto* tensor_data = tensor.data_ptr<float>();
  for (int e = 0, i = 0; e < H * W; ++e) {
    for (int c = 0; c < C; ++c, ++i) {
      tensor_data[i] = e;
    }
  }

  // quantize and check values
  Tensor q = at::native::quantize_per_channel_cpu(
      tensor, scales, zero_points, ch_axis, kQUInt8);
  auto* q_data = (uint8_t*)q.data_ptr<quint8>();
  for (int e = 0, i = 0; e < H * W; ++e) {
    for (int c = 0; c < C; ++c, ++i) {
      float inv_scale = 1.0f / static_cast<float>(scales[c].item<double>());
      int64_t zero_point = zero_points[c].item<int64_t>();
      // downsize qval to 255 if val is greater than max uint8_t value
      int qval = std::min<int>(zero_point + std::nearbyint(e * inv_scale), 255);
      ASSERT_EQ((int)q_data[i], qval);
    }
  }
}
