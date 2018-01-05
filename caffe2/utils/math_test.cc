/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "caffe2/core/blob.h"
#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

constexpr float kOne = 1.0f;
constexpr float kPointFive = 0.5f;
constexpr float kZero = 0.0f;

class GemmTest : public testing::TestWithParam<testing::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    cpu_context_ = make_unique<CPUContext>(option_);
    trans_X_ = std::get<0>(GetParam());
    trans_W_ = std::get<1>(GetParam());
    X_.Resize(trans_X_ ? std::vector<int>{5, 10} : std::vector<int>{10, 5});
    W_.Resize(trans_W_ ? std::vector<int>{10, 6} : std::vector<int>{6, 10});
    Y_.Resize(std::vector<int>{5, 6});
    math::Set<float, CPUContext>(
        X_.size(), 1.0f, X_.template mutable_data<float>(), cpu_context_.get());
    math::Set<float, CPUContext>(
        W_.size(), 1.0f, W_.template mutable_data<float>(), cpu_context_.get());
  }

  void RunGemm(const float alpha, const float beta) {
    math::Gemm<float, CPUContext>(
        trans_X_ ? CblasTrans : CblasNoTrans,
        trans_W_ ? CblasTrans : CblasNoTrans,
        5,
        6,
        10,
        alpha,
        X_.template data<float>(),
        W_.template data<float>(),
        beta,
        Y_.template mutable_data<float>(),
        cpu_context_.get());
  }

  void VerifyResult(const float value) const {
    ASSERT_EQ(30, Y_.size());
    for (int i = 0; i < Y_.size(); ++i) {
      EXPECT_FLOAT_EQ(value, Y_.template data<float>()[i]);
    }
  }

  DeviceOption option_;
  std::unique_ptr<CPUContext> cpu_context_;
  TensorCPU X_;
  TensorCPU W_;
  TensorCPU Y_;
  bool trans_X_;
  bool trans_W_;
};

TEST_P(GemmTest, GemmFloatTest) {
  RunGemm(kOne, kZero);
  VerifyResult(10.0f);
  RunGemm(kOne, kPointFive);
  VerifyResult(15.0f);
  RunGemm(kPointFive, kOne);
  VerifyResult(20.0f);
}

INSTANTIATE_TEST_CASE_P(
    GemmTransXTransWParam,
    GemmTest,
    testing::Combine(testing::Bool(), testing::Bool()));

class GemmBatchedTest
    : public testing::TestWithParam<testing::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    cpu_context_ = make_unique<CPUContext>(option_);
    trans_X_ = std::get<0>(GetParam());
    trans_W_ = std::get<1>(GetParam());
    X_.Resize(
        trans_X_ ? std::vector<int>{3, 5, 10} : std::vector<int>{3, 10, 5});
    W_.Resize(
        trans_W_ ? std::vector<int>{3, 10, 6} : std::vector<int>{3, 6, 10});
    Y_.Resize(std::vector<int>{3, 5, 6});
    math::Set<float, CPUContext>(
        X_.size(), 1.0f, X_.template mutable_data<float>(), cpu_context_.get());
    math::Set<float, CPUContext>(
        W_.size(), 1.0f, W_.template mutable_data<float>(), cpu_context_.get());
  }

  void RunGemmBatched(const float alpha, const float beta) {
    math::GemmBatched<float, CPUContext>(
        trans_X_ ? CblasTrans : CblasNoTrans,
        trans_W_ ? CblasTrans : CblasNoTrans,
        3,
        5,
        6,
        10,
        alpha,
        X_.template data<float>(),
        W_.template data<float>(),
        beta,
        Y_.template mutable_data<float>(),
        cpu_context_.get());
  }

  void VerifyResult(const float value) const {
    ASSERT_EQ(90, Y_.size());
    for (int i = 0; i < Y_.size(); ++i) {
      EXPECT_FLOAT_EQ(value, Y_.template data<float>()[i]);
    }
  }

  DeviceOption option_;
  std::unique_ptr<CPUContext> cpu_context_;
  TensorCPU X_;
  TensorCPU W_;
  TensorCPU Y_;
  bool trans_X_;
  bool trans_W_;
};

TEST_P(GemmBatchedTest, GemmBatchedFloatTest) {
  RunGemmBatched(kOne, kZero);
  VerifyResult(10.0f);
  RunGemmBatched(kOne, kPointFive);
  VerifyResult(15.0f);
  RunGemmBatched(kPointFive, kOne);
  VerifyResult(20.0f);
}

INSTANTIATE_TEST_CASE_P(
    GemmTransXTransWParam,
    GemmBatchedTest,
    testing::Combine(testing::Bool(), testing::Bool()));

} // namespace

TEST(MathTest, GemvNoTrans) {
  DeviceOption option;
  CPUContext cpu_context(option);
  TensorCPU A(std::vector<int>{5, 10});
  TensorCPU X(std::vector<int>{10});
  TensorCPU Y(std::vector<int>{5});
  EXPECT_EQ(A.size(), 50);
  EXPECT_EQ(X.size(), 10);
  math::Set<float, CPUContext>(
      A.size(), 1, A.mutable_data<float>(), &cpu_context);
  math::Set<float, CPUContext>(
      X.size(), 1, X.mutable_data<float>(), &cpu_context);
  EXPECT_EQ(Y.size(), 5);
  for (int i = 0; i < A.size(); ++i) {
    CHECK_EQ(A.data<float>()[i], 1);
  }
  for (int i = 0; i < X.size(); ++i) {
    CHECK_EQ(X.data<float>()[i], 1);
  }

  math::Gemv<float, CPUContext>(
      CblasNoTrans,
      5,
      10,
      kOne,
      A.data<float>(),
      X.data<float>(),
      kZero,
      Y.mutable_data<float>(),
      &cpu_context);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 10) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUContext>(
      CblasNoTrans,
      5,
      10,
      kOne,
      A.data<float>(),
      X.data<float>(),
      kPointFive,
      Y.mutable_data<float>(),
      &cpu_context);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 15) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUContext>(
      CblasNoTrans,
      5,
      10,
      kPointFive,
      A.data<float>(),
      X.data<float>(),
      kOne,
      Y.mutable_data<float>(),
      &cpu_context);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 20) << i;
  }
}

TEST(MathTest, GemvTrans) {
  DeviceOption option;
  CPUContext cpu_context(option);
  TensorCPU A(std::vector<int>{6, 10});
  TensorCPU X(std::vector<int>{6});
  TensorCPU Y(std::vector<int>{10});
  EXPECT_EQ(A.size(), 60);
  EXPECT_EQ(X.size(), 6);
  math::Set<float, CPUContext>(
      A.size(), 1, A.mutable_data<float>(), &cpu_context);
  math::Set<float, CPUContext>(
      X.size(), 1, X.mutable_data<float>(), &cpu_context);
  EXPECT_EQ(Y.size(), 10);
  for (int i = 0; i < A.size(); ++i) {
    CHECK_EQ(A.data<float>()[i], 1);
  }
  for (int i = 0; i < X.size(); ++i) {
    CHECK_EQ(X.data<float>()[i], 1);
  }

  math::Gemv<float, CPUContext>(
      CblasTrans,
      6,
      10,
      kOne,
      A.data<float>(),
      X.data<float>(),
      kZero,
      Y.mutable_data<float>(),
      &cpu_context);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 6) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUContext>(
      CblasTrans,
      6,
      10,
      kOne,
      A.data<float>(),
      X.data<float>(),
      kPointFive,
      Y.mutable_data<float>(),
      &cpu_context);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 9) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUContext>(
      CblasTrans,
      6,
      10,
      kPointFive,
      A.data<float>(),
      X.data<float>(),
      kOne,
      Y.mutable_data<float>(),
      &cpu_context);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 12) << i;
  }
}

using convert::cpu_float2half_rn;
using convert::cpu_half2float;

TEST(MathTest, FloatToHalfConversion) {
  float a = 1.0f;
  float b = 1.75f;
  float c = 128.125f;

  float converted_a = cpu_half2float(cpu_float2half_rn(a));
  float converted_b = cpu_half2float(cpu_float2half_rn(b));
  float converted_c = cpu_half2float(cpu_float2half_rn(c));

  CHECK_EQ(a, converted_a);
  CHECK_EQ(b, converted_b);
  CHECK_EQ(c, converted_c);
}

TEST(MathTest, TranposeTest) {
  DeviceOption option;
  CPUContext cpu_context(option);

  {
    // Test for 1D transpose.
    TensorCPU X(std::vector<int>{3});
    TensorCPU Y(std::vector<int>{3});
    for (int i = 0; i < 3; ++i) {
      X.mutable_data<float>()[i] = static_cast<float>(i + 1);
    }
    math::Transpose<float, CPUContext>(
        X.dims(),
        Y.dims(),
        {0},
        X.data<float>(),
        Y.mutable_data<float>(),
        &cpu_context);
    for (int i = 0; i < 3; ++i) {
      EXPECT_FLOAT_EQ(static_cast<float>(i + 1), Y.data<float>()[i]);
    }
  }

  {
    // Test for 2D transpose.
    TensorCPU X(std::vector<int>{2, 3});
    TensorCPU Y(std::vector<int>{3, 2});
    for (int i = 0; i < 6; ++i) {
      X.mutable_data<float>()[i] = static_cast<float>(i + 1);
    }
    math::Transpose<float, CPUContext>(
        X.dims(),
        Y.dims(),
        {1, 0},
        X.data<float>(),
        Y.mutable_data<float>(),
        &cpu_context);
    const std::vector<float> expected_output = {
        1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    for (int i = 0; i < 6; ++i) {
      EXPECT_FLOAT_EQ(expected_output[i], Y.data<float>()[i]);
    }
  }

  {
    // Test for 3D transpose.
    TensorCPU X(std::vector<int>{2, 2, 2});
    TensorCPU Y(std::vector<int>{2, 2, 2});
    for (int i = 0; i < 8; ++i) {
      X.mutable_data<float>()[i] = static_cast<float>(i + 1);
    }
    math::Transpose<float, CPUContext>(
        X.dims(),
        Y.dims(),
        {1, 2, 0},
        X.data<float>(),
        Y.mutable_data<float>(),
        &cpu_context);
    const std::vector<float> expected_output1 = {
        1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f};
    for (int i = 0; i < 8; ++i) {
      EXPECT_FLOAT_EQ(expected_output1[i], Y.data<float>()[i]);
    }

    math::Set<float, CPUContext>(
        Y.size(), 0.0f, Y.mutable_data<float>(), &cpu_context);
    math::Transpose<float, CPUContext>(
        X.dims(),
        Y.dims(),
        {1, 0, 2},
        X.data<float>(),
        Y.mutable_data<float>(),
        &cpu_context);
    const std::vector<float> expected_output2 = {
        1.0f, 2.0f, 5.0f, 6.0f, 3.0f, 4.0f, 7.0f, 8.0f};
    for (int i = 0; i < 8; ++i) {
      EXPECT_FLOAT_EQ(expected_output2[i], Y.data<float>()[i]);
    }
  }
}

} // namespace caffe2
