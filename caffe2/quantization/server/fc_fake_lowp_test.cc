#include <array>
#include <bitset>
#include <iomanip>
#include <limits>
#include <random>

#include <gtest/gtest.h>
#include "caffe2/core/logging.h"
#include "fully_connected_fake_lowp_op.h"

constexpr size_t sz = 10000;
using array = std::array<float, sz>;

constexpr size_t sz_test = 10;
using testarray = std::array<float, sz_test>;

bool isclose(float x, float y) {
  if (x == y) {
    return true;
  }
  LOG(INFO) << "Error range: " << fabs((x - y) / x);
  return fabs(x - y) < 1e-2 * fabs(x);
}

bool isrelclose(float x, float y) {
  if (x == y) {
    return true;
  }
  float relerr = float(1.0) / float(1 << 7);
  LOG(INFO) << "Relative error range: " << fabs((x - y) / x) << " " << relerr;

  return fabs((x - y) / x) < relerr;
}

template <std::size_t N>
double mse(std::array<float, N>& a1, std::array<float, N>& a2) {
  double total = 0.0;
  for (auto i = 0; i < N; i++) {
    auto diff = a1[i] - a2[i];
    total += diff * diff;
  }
  return sqrt(total) / N;
}

// minimum representable value of (10 bit mantissa) fp16=6e-5
// minimum representable value of (7 bit mantissa) bfp16=5e-4
static void test_case(const float v, const float v_fp16, const float v_bfp16) {
  std::array<float, 1> input, output1, output2, output3, output4, output5;
  input[0] = v;
  caffe2::fp32_to_fp16(input.data(), 1, output1.data());
  caffe2::fp32_to_bfp16(input.data(), 1, output2.data());
  caffe2::fp32_to_bfp24(input.data(), 1, output3.data());
  caffe2::fp32_to_bfp14(input.data(), 1, output4.data());
  caffe2::fp32_to_bfp16_round(input.data(), 1, output5.data());

  if (!std::isinf(mse<1>(input, output1))) {
    // If fp16 doesn't overshoot, the acurracy should be always better
    // than the one where we do truncation
    EXPECT_TRUE(mse<1>(input, output2) >= mse<1>(input, output1));
  }

  LOG(INFO) << std::hex << std::showbase << *(int*)(&input[0]) << " "
            << std::setprecision(20) << input[0];
  LOG(INFO) << std::hex << std::showbase << *(int*)(&output1[0]) << " "
            << std::setprecision(20) << output1[0];
  LOG(INFO) << std::hex << std::showbase << *(int*)(&output2[0]) << " "
            << std::setprecision(20) << output2[0];
  LOG(INFO) << std::hex << std::showbase << *(int*)(&output3[0]) << " "
            << std::setprecision(20) << output3[0];
  LOG(INFO) << std::hex << std::showbase << *(int*)(&output4[0]) << " "
            << std::setprecision(20) << output4[0];
  LOG(INFO) << std::hex << std::showbase << *(int*)(&output5[0]) << " "
            << std::setprecision(20) << output5[0];
  EXPECT_TRUE(isclose(output1[0], v_fp16));
  EXPECT_TRUE(isclose(output2[0], v_bfp16));
  EXPECT_TRUE(isclose(output5[0], output2[0]));
  EXPECT_TRUE(isrelclose(output5[0], output2[0]));
}

static void
test_vector_case(const float v, const float v_fp16, const float v_bfp16) {
  testarray vv, vfp16, vbfp16, vbfp16s, vbfp24, vbfp14, vbfp16r;

  int i;
  for (i = 0; i < sz_test; i++) {
    vv[i] = v;
  }

  caffe2::fp32_to_fp16(vv.data(), sz_test, vfp16.data());
  caffe2::fp32_to_bfp16(vv.data(), sz_test, vbfp16.data());
  caffe2::fp32_to_bfp16_scalar(vv.data(), sz_test, vbfp16s.data());
  caffe2::fp32_to_bfp24(vv.data(), sz_test, vbfp24.data());
  caffe2::fp32_to_bfp14(vv.data(), sz_test, vbfp14.data());
  caffe2::fp32_to_bfp16_round(vv.data(), sz_test, vbfp16r.data());

  LOG(INFO) << "vector " << vv[0] << " " << vfp16[0] << " " << vbfp16[0] << " "
            << vbfp24[0] << " " << vbfp14[0];

  for (auto ii = 0; i < sz_test; i++) {
    EXPECT_TRUE(isclose(vfp16[i], v_fp16));
    EXPECT_TRUE(isclose(vbfp16[i], v_bfp16));
    EXPECT_TRUE(vbfp16[i] == vbfp16s[i]);
    EXPECT_TRUE(isclose(vbfp16[i], vbfp16r[i]));
  }
}

TEST(FP16Quant, fp32_to_fp16) {
  array input, output1, output2;

  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, 1);

  test_case(10.0, 10.0, 10.0);
  test_vector_case(10.0, 10.0, 10.0);
  test_case(65000, 64992, 64768);
  test_vector_case(65000, 64992, 64768);
  test_case(1.2345678910, 1.234375, 1.234375);
  test_vector_case(1.2345678910, 1.234375, 1.234375);
  test_case(3.3333333, 3.333984375, 3.328125);
  test_vector_case(3.3333333, 3.333984375, 3.328125);
  test_case(123e10, std::numeric_limits<float>::infinity(), 122.8360646656e10);
  test_vector_case(
      123e10, std::numeric_limits<float>::infinity(), 122.8360646656e10);
  test_case(65504, 65504, 65280);
  test_vector_case(65504, 65506, 65280);
  test_case(65504 * 4, std::numeric_limits<float>::infinity(), 261120);
  test_vector_case(65504 * 4, std::numeric_limits<float>::infinity(), 261120);
  test_case(3.402823466e-37, 0, 3.3942e-37);
  test_vector_case(3.402823466e-37, 0, 3.3942e-37);

  for (auto i = 0; i < sz; i++) {
    input[i] = dist(e2) * 1e8;
  }

  caffe2::fp32_to_fp16(input.data(), input.size(), output1.data());
  caffe2::fp32_to_bfp16(input.data(), input.size(), output2.data());

  LOG(INFO) << "None " << mse<sz>(input, input);
  LOG(INFO) << "FP16 " << mse<sz>(input, output1);
  LOG(INFO) << "BFP16 " << mse<sz>(input, output2);

  LOG(INFO) << std::hex << std::showbase << *(int*)(&input[0]) << " "
            << input[0];
  LOG(INFO) << std::hex << std::showbase << *(int*)(&output1[0]) << " "
            << output1[0];
  LOG(INFO) << std::hex << std::showbase << *(int*)(&output2[0]) << " "
            << output2[0];
}
