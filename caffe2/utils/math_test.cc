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

namespace caffe2 {

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

class GemmBatchedTest
    : public testing::TestWithParam<testing::tuple<bool, bool>> {
 protected:
  void SetUp() override {
    cpu_context_ = make_unique<CPUContext>(option_);
    X_.Resize(std::vector<int64_t>{3, 5, 10});
    W_.Resize(std::vector<int64_t>{3, 6, 10});
    Y_.Resize(std::vector<int64_t>{3, 5, 6});
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

  DeviceOption option_;
  std::unique_ptr<CPUContext> cpu_context_;
  Tensor X_{CPU};
  Tensor W_{CPU};
  Tensor Y_{CPU};
  bool trans_X_;
  bool trans_W_;
};

TEST_P(GemmBatchedTest, GemmBatchedFloatTest) {
  RunGemmBatched(1.0f, 0.0f);
  VerifyOutput(10.0f);
  RunGemmBatched(1.0f, 0.5f);
  VerifyOutput(15.0f);
  RunGemmBatched(0.5f, 1.0f);
  VerifyOutput(20.0f);
}

TEST_P(GemmBatchedTest, GemmStridedBatchedFloatTest) {
  RunGemmStridedBatched(1.0f, 0.0f);
  VerifyOutput(10.0f);
  RunGemmStridedBatched(1.0f, 0.5f);
  VerifyOutput(15.0f);
  RunGemmStridedBatched(0.5f, 1.0f);
  VerifyOutput(20.0f);
}

INSTANTIATE_TEST_CASE_P(
    GemmBatchedTrans,
    GemmBatchedTest,
    testing::Combine(testing::Bool(), testing::Bool()));

} // namespace

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

class ReduceTensorTest : public testing::Test {
 protected:
  void SetUp() override {
    cpu_context_ = make_unique<CPUContext>(option_);
  }

  template <class ReduceFunc>
  void RunRedcueTensorTest(
      const ReduceFunc& reduce_func,
      const std::vector<int>& X_dims,
      const std::vector<int>& axes,
      const std::vector<float>& X_data,
      const std::vector<float>& Y_data) {
    std::vector<int> Y_dims = X_dims;
    for (const int axis : axes) {
      Y_dims[axis] = 1;
    }
    X_.Resize(X_dims);
    Y_.Resize(Y_dims);
    ASSERT_EQ(X_data.size(), X_.numel());
    cpu_context_->CopyFromCPU<float>(
        X_data.size(), X_data.data(), X_.mutable_data<float>());
    reduce_func(
        X_dims.size(),
        X_dims.data(),
        axes.size(),
        axes.data(),
        1.0f,
        X_.data<float>(),
        Y_.mutable_data<float>(),
        cpu_context_.get());
    ASSERT_EQ(Y_data.size(), Y_.numel());
    for (int i = 0; i < Y_.numel(); ++i) {
      EXPECT_FLOAT_EQ(Y_data[i], Y_.data<float>()[i]);
    }
  }

  DeviceOption option_;
  std::unique_ptr<CPUContext> cpu_context_;
  Tensor X_{CPU};
  Tensor Y_{CPU};
};

TEST_F(ReduceTensorTest, ReduceMinTest) {
  const auto& reduce_min = [](const int num_dims,
                              const int* dims,
                              const int num_axes,
                              const int* axes,
                              const float alpha,
                              const float* X,
                              float* Y,
                              CPUContext* context) {
    return math::ReduceMin<float, CPUContext>(
        num_dims, dims, num_axes, axes, alpha, X, Y, context);
  };
  // Test for 1D tensor.
  RunRedcueTensorTest(reduce_min, {3}, {0}, {1.0f, 2.0f, 3.0f}, {1.0f});

  // Test for 2D Tensor.
  RunRedcueTensorTest(
      reduce_min,
      {2, 3},
      {1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {1.0f, 4.0f});
  RunRedcueTensorTest(
      reduce_min,
      {2, 3},
      {0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {1.0f, 2.0f, 3.0f});
  RunRedcueTensorTest(
      reduce_min, {2, 3}, {0, 1}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {1.0f});

  // Test for 3D tensor.
  RunRedcueTensorTest(
      reduce_min,
      {2, 2, 2},
      {1, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {1.0f, 5.0f});
  RunRedcueTensorTest(
      reduce_min,
      {2, 2, 2},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {1.0f, 2.0f});
  RunRedcueTensorTest(
      reduce_min,
      {2, 2, 2},
      {0, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {1.0f, 3.0f});
}

TEST_F(ReduceTensorTest, ReduceMaxTest) {
  const auto& reduce_max = [](const int num_dims,
                              const int* dims,
                              const int num_axes,
                              const int* axes,
                              const float alpha,
                              const float* X,
                              float* Y,
                              CPUContext* context) {
    return math::ReduceMax<float, CPUContext>(
        num_dims, dims, num_axes, axes, alpha, X, Y, context);
  };
  // Test for 1D tensor.
  RunRedcueTensorTest(reduce_max, {3}, {0}, {1.0f, 2.0f, 3.0f}, {3.0f});

  // Test for 2D Tensor.
  RunRedcueTensorTest(
      reduce_max,
      {2, 3},
      {1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {3.0f, 6.0f});
  RunRedcueTensorTest(
      reduce_max,
      {2, 3},
      {0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {4.0f, 5.0f, 6.0f});
  RunRedcueTensorTest(
      reduce_max, {2, 3}, {0, 1}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {6.0f});

  // Test for 3D tensor.
  RunRedcueTensorTest(
      reduce_max,
      {2, 2, 2},
      {1, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {4.0f, 8.0f});
  RunRedcueTensorTest(
      reduce_max,
      {2, 2, 2},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {7.0f, 8.0f});
  RunRedcueTensorTest(
      reduce_max,
      {2, 2, 2},
      {0, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {6.0f, 8.0f});
}

TEST_F(ReduceTensorTest, ReduceSumTest) {
  // Test for 1D tensor.
  RunRedcueTensorTest(
      math::ReduceSum<float, CPUContext>, {3}, {0}, {1.0f, 2.0f, 3.0f}, {6.0f});

  // Test for 2D Tensor.
  RunRedcueTensorTest(
      math::ReduceSum<float, CPUContext>,
      {2, 3},
      {1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {6.0f, 15.0f});
  RunRedcueTensorTest(
      math::ReduceSum<float, CPUContext>,
      {2, 3},
      {0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {5.0f, 7.0f, 9.0f});
  RunRedcueTensorTest(
      math::ReduceSum<float, CPUContext>,
      {2, 3},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {21.0f});

  // Test for 3D tensor.
  RunRedcueTensorTest(
      math::ReduceSum<float, CPUContext>,
      {2, 2, 2},
      {1, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {10.0f, 26.0f});
  RunRedcueTensorTest(
      math::ReduceSum<float, CPUContext>,
      {2, 2, 2},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {16.0f, 20.0f});
  RunRedcueTensorTest(
      math::ReduceSum<float, CPUContext>,
      {2, 2, 2},
      {0, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {14.0f, 22.0f});
}

TEST_F(ReduceTensorTest, ReduceMeanTest) {
  // Test for 1D tensor.
  RunRedcueTensorTest(
      math::ReduceMean<float, CPUContext>,
      {3},
      {0},
      {1.0f, 2.0f, 3.0f},
      {2.0f});

  // Test for 2D Tensor.
  RunRedcueTensorTest(
      math::ReduceMean<float, CPUContext>,
      {2, 3},
      {1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {2.0f, 5.0f});
  RunRedcueTensorTest(
      math::ReduceMean<float, CPUContext>,
      {2, 3},
      {0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {2.5f, 3.5f, 4.5f});
  RunRedcueTensorTest(
      math::ReduceMean<float, CPUContext>,
      {2, 3},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {3.5f});

  // Test for 3D tensor.
  RunRedcueTensorTest(
      math::ReduceMean<float, CPUContext>,
      {2, 2, 2},
      {1, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {2.5f, 6.5f});
  RunRedcueTensorTest(
      math::ReduceMean<float, CPUContext>,
      {2, 2, 2},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {4.0f, 5.0f});
  RunRedcueTensorTest(
      math::ReduceMean<float, CPUContext>,
      {2, 2, 2},
      {0, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {3.5f, 5.5f});
}

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
    X_.Resize(X_dims);
    Y_.Resize(Y_dims);
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
    for (int i = 0; i < Y_data.size(); ++i) {
      EXPECT_FLOAT_EQ(Y_data[i], Y_.data<float>()[i]);
    }
  }

  DeviceOption option_;
  std::unique_ptr<CPUContext> cpu_context_;

  Tensor X_{CPU};
  Tensor Y_{CPU};
};

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

class MomentsTest : public testing::Test {
 protected:
  void SetUp() override {
    cpu_context_ = make_unique<CPUContext>(option_);
  }

  void RunMomentsTest(
      const std::vector<int>& X_dims,
      const std::vector<int>& axes,
      const std::vector<float>& X_data,
      const std::vector<float>& mean_data,
      const std::vector<float>& variance_data) {
    const int ndim = X_dims.size();
    std::vector<int> Y_dims = X_dims;
    for (const int axis : axes) {
      Y_dims[axis] = 1;
    }
    X_.Resize(X_dims);
    mean_.Resize(Y_dims);
    variance_.Resize(Y_dims);
    ASSERT_EQ(X_data.size(), X_.numel());
    cpu_context_->CopyFromCPU<float>(
        X_data.size(), X_data.data(), X_.mutable_data<float>());
    math::Moments<float, CPUContext>(
        X_dims.size(),
        X_dims.data(),
        axes.size(),
        axes.data(),
        X_.data<float>(),
        mean_.mutable_data<float>(),
        variance_.mutable_data<float>(),
        cpu_context_.get());
    ASSERT_EQ(mean_data.size(), mean_.numel());
    for (int i = 0; i < mean_data.size(); ++i) {
      EXPECT_FLOAT_EQ(mean_data[i], mean_.data<float>()[i]);
    }
    ASSERT_EQ(variance_data.size(), variance_.numel());
    for (int i = 0; i < variance_data.size(); ++i) {
      EXPECT_NEAR(variance_data[i], variance_.data<float>()[i], kEps);
    }
  }

  DeviceOption option_;
  std::unique_ptr<CPUContext> cpu_context_;

  Tensor X_{CPU};
  Tensor mean_{CPU};
  Tensor variance_{CPU};
};

TEST_F(MomentsTest, MomentsFloatTest) {
  // Test for 1D tensor.
  RunMomentsTest({3}, {0}, {1.0f, 2.0f, 3.0f}, {2.0f}, {2.0f / 3.0f});

  // Test for 2D Tensor.
  RunMomentsTest(
      {2, 3},
      {1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {2.0f, 5.0f},
      {2.0f / 3.0f, 2.0f / 3.0f});
  RunMomentsTest(
      {2, 3},
      {0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {2.5f, 3.5f, 4.5f},
      {2.25f, 2.25f, 2.25f});
  RunMomentsTest(
      {2, 3},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {3.5f},
      {35.0f / 12.0f});

  // Test for 3D tensor.
  RunMomentsTest(
      {2, 2, 2},
      {1, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {2.5f, 6.5f},
      {1.25, 1.25});
  RunMomentsTest(
      {2, 2, 2},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {4.0f, 5.0f},
      {5.0f, 5.0f});
  RunMomentsTest(
      {2, 2, 2},
      {0, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {3.5f, 5.5f},
      {4.25, 4.25});
}

class TransposeTest : public testing::Test {
 protected:
  void SetUp() override {
    cpu_context_ = make_unique<CPUContext>(option_);
  }

  void RunTransposeTest(
      const std::vector<int>& X_dims,
      const std::vector<int>& axes,
      const std::vector<float>& X_data,
      const std::vector<float>& Y_data) {
    const int ndim = X_dims.size();
    std::vector<int> Y_dims(ndim);
    for (int i = 0; i < ndim; ++i) {
      Y_dims[i] = X_dims[axes[i]];
    }
    X_.Resize(X_dims);
    Y_.Resize(Y_dims);
    ASSERT_EQ(X_data.size(), X_.numel());
    cpu_context_->CopyFromCPU<float>(
        X_data.size(), X_data.data(), X_.mutable_data<float>());
    math::Transpose<float, CPUContext>(
        X_dims.size(),
        X_dims.data(),
        axes.data(),
        X_.data<float>(),
        Y_.mutable_data<float>(),
        cpu_context_.get());
    ASSERT_EQ(Y_data.size(), Y_.numel());
    for (int i = 0; i < Y_.numel(); ++i) {
      EXPECT_FLOAT_EQ(Y_data[i], Y_.data<float>()[i]);
    }
  }

  DeviceOption option_;
  std::unique_ptr<CPUContext> cpu_context_;

  Tensor X_{CPU};
  Tensor Y_{CPU};
};

TEST_F(TransposeTest, TransposeFloatTest) {
  // Test for 1D transpose.
  RunTransposeTest({3}, {0}, {1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 3.0f});

  // Test for 2D transpose.
  RunTransposeTest(
      {2, 3},
      {1, 0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});

  // Test for 3D transpose.
  RunTransposeTest(
      {2, 2, 2},
      {1, 2, 0},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f});
  RunTransposeTest(
      {2, 2, 2},
      {1, 0, 2},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {1.0f, 2.0f, 5.0f, 6.0f, 3.0f, 4.0f, 7.0f, 8.0f});
}

} // namespace

} // namespace caffe2
