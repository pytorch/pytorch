#include <array>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "caffe2/core/blob.h"
#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

#include <c10/util/irange.h>

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MathTest, GemmNoTransNoTrans) {
  DeviceOption option;
  CPUContext cpu_context(option);
  Tensor X(std::vector<int>{5, 10}, CPU);
  Tensor W(std::vector<int>{10, 6}, CPU);
  Tensor Y(std::vector<int>{5, 6}, CPU);
  EXPECT_EQ(X.numel(), 50);
  EXPECT_EQ(W.numel(), 60);
  math::Set<float, CPUContext>(
      X.numel(), 1, X.mutable_data<float>(), &cpu_context);
  math::Set<float, CPUContext>(
      W.numel(), 1, W.mutable_data<float>(), &cpu_context);
  EXPECT_EQ(Y.numel(), 30);
  for (int i = 0; i < X.numel(); ++i) {
    CHECK_EQ(X.data<float>()[i], 1);
  }
  for (int i = 0; i < W.numel(); ++i) {
    CHECK_EQ(W.data<float>()[i], 1);
  }

  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
  math::Gemm<float, CPUContext>(
      CblasNoTrans,
      CblasNoTrans,
      5,
      6,
      10,
      kOne,
      X.data<float>(),
      W.data<float>(),
      kZero,
      Y.mutable_data<float>(),
      &cpu_context);
  EXPECT_EQ(Y.numel(), 30);
  for (int i = 0; i < Y.numel(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 10) << i;
  }
  // Test Accumulate
  math::Gemm<float, CPUContext>(
      CblasNoTrans,
      CblasNoTrans,
      5,
      6,
      10,
      kOne,
      X.data<float>(),
      W.data<float>(),
      kPointFive,
      Y.mutable_data<float>(),
      &cpu_context);
  EXPECT_EQ(Y.numel(), 30);
  for (int i = 0; i < Y.numel(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 15) << i;
  }
  // Test Accumulate
  math::Gemm<float, CPUContext>(
      CblasNoTrans,
      CblasNoTrans,
      5,
      6,
      10,
      kPointFive,
      X.data<float>(),
      W.data<float>(),
      kOne,
      Y.mutable_data<float>(),
      &cpu_context);
  EXPECT_EQ(Y.numel(), 30);
  for (int i = 0; i < Y.numel(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 20) << i;
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MathTest, GemmNoTransTrans) {
  DeviceOption option;
  CPUContext cpu_context(option);
  Tensor X(std::vector<int>{5, 10}, CPU);
  Tensor W(std::vector<int>{6, 10}, CPU);
  Tensor Y(std::vector<int>{5, 6}, CPU);
  EXPECT_EQ(X.numel(), 50);
  EXPECT_EQ(W.numel(), 60);
  math::Set<float, CPUContext>(
      X.numel(), 1, X.mutable_data<float>(), &cpu_context);
  math::Set<float, CPUContext>(
      W.numel(), 1, W.mutable_data<float>(), &cpu_context);
  EXPECT_EQ(Y.numel(), 30);
  for (int i = 0; i < X.numel(); ++i) {
    CHECK_EQ(X.data<float>()[i], 1);
  }
  for (int i = 0; i < W.numel(); ++i) {
    CHECK_EQ(W.data<float>()[i], 1);
  }

  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
  math::Gemm<float, CPUContext>(
      CblasNoTrans,
      CblasTrans,
      5,
      6,
      10,
      kOne,
      X.data<float>(),
      W.data<float>(),
      kZero,
      Y.mutable_data<float>(),
      &cpu_context);
  EXPECT_EQ(Y.numel(), 30);
  for (int i = 0; i < Y.numel(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 10) << i;
  }
  // Test Accumulate
  math::Gemm<float, CPUContext>(
      CblasNoTrans,
      CblasTrans,
      5,
      6,
      10,
      kOne,
      X.data<float>(),
      W.data<float>(),
      kPointFive,
      Y.mutable_data<float>(),
      &cpu_context);
  EXPECT_EQ(Y.numel(), 30);
  for (int i = 0; i < Y.numel(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 15) << i;
  }
  math::Gemm<float, CPUContext>(
      CblasNoTrans,
      CblasTrans,
      5,
      6,
      10,
      kPointFive,
      X.data<float>(),
      W.data<float>(),
      kOne,
      Y.mutable_data<float>(),
      &cpu_context);
  EXPECT_EQ(Y.numel(), 30);
  for (int i = 0; i < Y.numel(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 20) << i;
  }
}

namespace {

constexpr float kEps = 1e-5;

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class GemmBatchedTest
    : public testing::TestWithParam<testing::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    cpu_context_ = make_unique<CPUContext>(option_);
    ReinitializeTensor(
        &X_, std::vector<int64_t>{3, 5, 10}, at::dtype<float>().device(CPU));
    ReinitializeTensor(
        &W_, std::vector<int64_t>{3, 6, 10}, at::dtype<float>().device(CPU));
    ReinitializeTensor(
        &Y_, std::vector<int64_t>{3, 5, 6}, at::dtype<float>().device(CPU));
    math::Set<float, CPUContext>(
        X_.numel(), 1, X_.mutable_data<float>(), cpu_context_.get());
    math::Set<float, CPUContext>(
        W_.numel(), 1, W_.mutable_data<float>(), cpu_context_.get());
    trans_X_ = std::get<0>(GetParam());
    trans_W_ = std::get<1>(GetParam());
  }

  void RunGemmBatched(const float alpha, const float beta) {
    const float* X_data = X_.template data<float>();
    const float* W_data = W_.template data<float>();
    float* Y_data = Y_.template mutable_data<float>();
    const int X_stride = 5 * 10;
    const int W_stride = 6 * 10;
    const int Y_stride = 5 * 6;
    std::array<const float*, 3> X_array = {
        X_data, X_data + X_stride, X_data + 2 * X_stride};
    std::array<const float*, 3> W_array = {
        W_data, W_data + W_stride, W_data + 2 * W_stride};
    std::array<float*, 3> Y_array = {
        Y_data, Y_data + Y_stride, Y_data + 2 * Y_stride};
    math::GemmBatched(
        trans_X_ ? CblasTrans : CblasNoTrans,
        trans_W_ ? CblasTrans : CblasNoTrans,
        3,
        5,
        6,
        10,
        alpha,
        X_array.data(),
        W_array.data(),
        beta,
        Y_array.data(),
        cpu_context_.get());
  }

  void RunGemmStridedBatched(const float alpha, const float beta) {
    const float* X_data = X_.template data<float>();
    const float* W_data = W_.template data<float>();
    float* Y_data = Y_.template mutable_data<float>();
    const int X_stride = 5 * 10;
    const int W_stride = 6 * 10;
    const int Y_stride = 5 * 6;
    math::GemmStridedBatched<float, CPUContext>(
        trans_X_ ? CblasTrans : CblasNoTrans,
        trans_W_ ? CblasTrans : CblasNoTrans,
        3,
        5,
        6,
        10,
        alpha,
        X_data,
        X_stride,
        W_data,
        W_stride,
        beta,
        Y_data,
        Y_stride,
        cpu_context_.get());
  }

  void VerifyOutput(const float value) const {
    for (int i = 0; i < Y_.numel(); ++i) {
      EXPECT_FLOAT_EQ(value, Y_.template data<float>()[i]);
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  DeviceOption option_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<CPUContext> cpu_context_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Tensor X_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Tensor W_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Tensor Y_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool trans_X_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool trans_W_;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_P(GemmBatchedTest, GemmBatchedFloatTest) {
  RunGemmBatched(1.0f, 0.0f);
  VerifyOutput(10.0f);
  RunGemmBatched(1.0f, 0.5f);
  VerifyOutput(15.0f);
  RunGemmBatched(0.5f, 1.0f);
  VerifyOutput(20.0f);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_P(GemmBatchedTest, GemmStridedBatchedFloatTest) {
  RunGemmStridedBatched(1.0f, 0.0f);
  VerifyOutput(10.0f);
  RunGemmStridedBatched(1.0f, 0.5f);
  VerifyOutput(15.0f);
  RunGemmStridedBatched(0.5f, 1.0f);
  VerifyOutput(20.0f);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
INSTANTIATE_TEST_CASE_P(
    GemmBatchedTrans,
    GemmBatchedTest,
    testing::Combine(testing::Bool(), testing::Bool()));

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MathTest, GemvNoTrans) {
  DeviceOption option;
  CPUContext cpu_context(option);
  Tensor A(std::vector<int>{5, 10}, CPU);
  Tensor X(std::vector<int>{10}, CPU);
  Tensor Y(std::vector<int>{5}, CPU);
  EXPECT_EQ(A.numel(), 50);
  EXPECT_EQ(X.numel(), 10);
  math::Set<float, CPUContext>(
      A.numel(), 1, A.mutable_data<float>(), &cpu_context);
  math::Set<float, CPUContext>(
      X.numel(), 1, X.mutable_data<float>(), &cpu_context);
  EXPECT_EQ(Y.numel(), 5);
  for (int i = 0; i < A.numel(); ++i) {
    CHECK_EQ(A.data<float>()[i], 1);
  }
  for (int i = 0; i < X.numel(); ++i) {
    CHECK_EQ(X.data<float>()[i], 1);
  }

  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
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
  for (int i = 0; i < Y.numel(); ++i) {
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
  for (int i = 0; i < Y.numel(); ++i) {
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
  for (int i = 0; i < Y.numel(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 20) << i;
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MathTest, GemvTrans) {
  DeviceOption option;
  CPUContext cpu_context(option);
  Tensor A(std::vector<int>{6, 10}, CPU);
  Tensor X(std::vector<int>{6}, CPU);
  Tensor Y(std::vector<int>{10}, CPU);
  EXPECT_EQ(A.numel(), 60);
  EXPECT_EQ(X.numel(), 6);
  math::Set<float, CPUContext>(
      A.numel(), 1, A.mutable_data<float>(), &cpu_context);
  math::Set<float, CPUContext>(
      X.numel(), 1, X.mutable_data<float>(), &cpu_context);
  EXPECT_EQ(Y.numel(), 10);
  for (int i = 0; i < A.numel(); ++i) {
    CHECK_EQ(A.data<float>()[i], 1);
  }
  for (int i = 0; i < X.numel(); ++i) {
    CHECK_EQ(X.data<float>()[i], 1);
  }

  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
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
  for (int i = 0; i < Y.numel(); ++i) {
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
  for (int i = 0; i < Y.numel(); ++i) {
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
  for (int i = 0; i < Y.numel(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 12) << i;
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(MathTest, FloatToHalfConversion) {
  float a = 1.0f;
  float b = 1.75f;
  float c = 128.125f;

  float converted_a = static_cast<float>(at::Half(a));
  float converted_b = static_cast<float>(at::Half(b));
  float converted_c = static_cast<float>(at::Half(c));

  CHECK_EQ(a, converted_a);
  CHECK_EQ(b, converted_b);
  CHECK_EQ(c, converted_c);
}

namespace {

class BroadcastTest : public testing::Test {
 protected:
  void SetUp() override {
    cpu_context_ = make_unique<CPUContext>(option_);
  }

  void RunBroadcastTest(
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const std::vector<float>& X_data,
      const std::vector<float>& Y_data) {
    std::vector<int64_t> X_dims_64;
    std::vector<int64_t> Y_dims_64;
    std::copy(X_dims.cbegin(), X_dims.cend(), std::back_inserter(X_dims_64));
    std::copy(Y_dims.cbegin(), Y_dims.cend(), std::back_inserter(Y_dims_64));
    ReinitializeTensor(&X_, X_dims_64, at::dtype<float>().device(CPU));
    ReinitializeTensor(&Y_, Y_dims_64, at::dtype<float>().device(CPU));
    ASSERT_EQ(X_data.size(), X_.numel());
    cpu_context_->CopyFromCPU<float>(
        X_data.size(), X_data.data(), X_.mutable_data<float>());
    math::Broadcast<float, CPUContext>(
        X_dims.size(),
        X_dims.data(),
        Y_dims.size(),
        Y_dims.data(),
        1.0f,
        X_.data<float>(),
        Y_.mutable_data<float>(),
        cpu_context_.get());
    ASSERT_EQ(Y_data.size(), Y_.numel());
    for (const auto i : c10::irange(Y_data.size())) {
      EXPECT_FLOAT_EQ(Y_data[i], Y_.data<float>()[i]);
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  DeviceOption option_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<CPUContext> cpu_context_;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Tensor X_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Tensor Y_;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(BroadcastTest, BroadcastFloatTest) {
  RunBroadcastTest({2}, {2}, {1.0f, 2.0f}, {1.0f, 2.0f});
  RunBroadcastTest({1}, {2}, {1.0f}, {1.0f, 1.0f});
  RunBroadcastTest({1}, {2, 2}, {1.0f}, {1.0f, 1.0f, 1.0f, 1.0f});
  RunBroadcastTest({2, 1}, {2, 2}, {1.0f, 2.0f}, {1.0f, 1.0f, 2.0f, 2.0f});
  RunBroadcastTest(
      {2, 1},
      {2, 2, 2},
      {1.0f, 2.0f},
      {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f});
}

class RandFixedSumTest : public testing::Test {
 protected:
  void SetUp() override {
    cpu_context_ = make_unique<CPUContext>(option_);
  }
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  DeviceOption option_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<CPUContext> cpu_context_;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST_F(RandFixedSumTest, UpperBound) {
  std::vector<int> l(20);
  math::RandFixedSum<int, CPUContext>(
      20, 1, 1000, 1000, l.data(), cpu_context_.get());
}

} // namespace

} // namespace caffe2
