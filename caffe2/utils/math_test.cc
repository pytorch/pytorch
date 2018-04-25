#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "caffe2/core/blob.h"
#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

TEST(MathTest, GemmNoTransNoTrans) {
  DeviceOption option;
  CPUContext cpu_context(option);
  TensorCPU X(std::vector<int>{5, 10});
  TensorCPU W(std::vector<int>{10, 6});
  TensorCPU Y(std::vector<int>{5, 6});
  EXPECT_EQ(X.size(), 50);
  EXPECT_EQ(W.size(), 60);
  math::Set<float, CPUContext>(
      X.size(), 1, X.mutable_data<float>(), &cpu_context);
  math::Set<float, CPUContext>(
      W.size(), 1, W.mutable_data<float>(), &cpu_context);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < X.size(); ++i) {
    CHECK_EQ(X.data<float>()[i], 1);
  }
  for (int i = 0; i < W.size(); ++i) {
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
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
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
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
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
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 20) << i;
  }
}

TEST(MathTest, GemmNoTransTrans) {
  DeviceOption option;
  CPUContext cpu_context(option);
  TensorCPU X(std::vector<int>{5, 10});
  TensorCPU W(std::vector<int>{6, 10});
  TensorCPU Y(std::vector<int>{5, 6});
  EXPECT_EQ(X.size(), 50);
  EXPECT_EQ(W.size(), 60);
  math::Set<float, CPUContext>(
      X.size(), 1, X.mutable_data<float>(), &cpu_context);
  math::Set<float, CPUContext>(
      W.size(), 1, W.mutable_data<float>(), &cpu_context);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < X.size(); ++i) {
    CHECK_EQ(X.data<float>()[i], 1);
  }
  for (int i = 0; i < W.size(); ++i) {
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
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
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
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
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
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
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
    X_.Resize(std::vector<TIndex>{3, 5, 10});
    W_.Resize(std::vector<TIndex>{3, 6, 10});
    Y_.Resize(std::vector<TIndex>{3, 5, 6});
    math::Set<float, CPUContext>(
        X_.size(), 1, X_.mutable_data<float>(), cpu_context_.get());
    math::Set<float, CPUContext>(
        W_.size(), 1, W_.mutable_data<float>(), cpu_context_.get());
    trans_X_ = std::get<0>(GetParam());
    trans_W_ = std::get<1>(GetParam());
  }

  void RunGemmBatched(const float alpha, const float beta) {
    math::GemmBatched(
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

  void VerifyOutput(const float value) const {
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
  RunGemmBatched(1.0f, 0.0f);
  VerifyOutput(10.0f);
  RunGemmBatched(1.0f, 0.5f);
  VerifyOutput(15.0f);
  RunGemmBatched(0.5f, 1.0f);
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
    ASSERT_EQ(X_data.size(), X_.size());
    cpu_context_->Copy<float, CPUContext, CPUContext>(
        X_data.size(), X_data.data(), X_.mutable_data<float>());
    reduce_func(
        X_dims.size(),
        X_dims.data(),
        axes.size(),
        axes.data(),
        X_.data<float>(),
        Y_.mutable_data<float>(),
        cpu_context_.get());
    ASSERT_EQ(Y_data.size(), Y_.size());
    for (int i = 0; i < Y_.size(); ++i) {
      EXPECT_FLOAT_EQ(Y_data[i], Y_.data<float>()[i]);
    }
  }

  DeviceOption option_;
  std::unique_ptr<CPUContext> cpu_context_;
  TensorCPU X_;
  TensorCPU Y_;
};

TEST_F(ReduceTensorTest, ReduceMinTest) {
  const auto& reduce_min = [](const int num_dims,
                              const int* dims,
                              const int num_axes,
                              const int* axes,
                              const float* X,
                              float* Y,
                              CPUContext* context) {
    return math::ReduceMin<float, CPUContext>(
        num_dims, dims, num_axes, axes, X, Y, context);
  };
  // Test for 1D tensor.
  RunRedcueTensorTest(
      reduce_min,
      {3},
      {0},
      {1.0f, 2.0f, 3.0f},
      {1.0f});

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
      reduce_min,
      {2, 3},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {1.0f});

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
                              const float* X,
                              float* Y,
                              CPUContext* context) {
    return math::ReduceMax<float, CPUContext>(
        num_dims, dims, num_axes, axes, X, Y, context);
  };
  // Test for 1D tensor.
  RunRedcueTensorTest(
      reduce_max,
      {3},
      {0},
      {1.0f, 2.0f, 3.0f},
      {3.0f});

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
      reduce_max,
      {2, 3},
      {0, 1},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {6.0f});

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
      math::ReduceSum<float, CPUContext>,
      {3},
      {0},
      {1.0f, 2.0f, 3.0f},
      {6.0f});

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
    ASSERT_EQ(X_data.size(), X_.size());
    cpu_context_->Copy<float, CPUContext, CPUContext>(
        X_data.size(), X_data.data(), X_.mutable_data<float>());
    math::Broadcast<float, CPUContext>(
        X_dims.size(),
        X_dims.data(),
        Y_dims.size(),
        Y_dims.data(),
        X_.data<float>(),
        Y_.mutable_data<float>(),
        cpu_context_.get());
    ASSERT_EQ(Y_data.size(), Y_.size());
    for (int i = 0; i < Y_data.size(); ++i) {
      EXPECT_FLOAT_EQ(Y_data[i], Y_.data<float>()[i]);
    }
  }

  DeviceOption option_;
  std::unique_ptr<CPUContext> cpu_context_;

  TensorCPU X_;
  TensorCPU Y_;
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
    ASSERT_EQ(X_data.size(), X_.size());
    cpu_context_->Copy<float, CPUContext, CPUContext>(
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
    ASSERT_EQ(mean_data.size(), mean_.size());
    for (int i = 0; i < mean_data.size(); ++i) {
      EXPECT_FLOAT_EQ(mean_data[i], mean_.data<float>()[i]);
    }
    ASSERT_EQ(variance_data.size(), variance_.size());
    for (int i = 0; i < variance_data.size(); ++i) {
      EXPECT_NEAR(variance_data[i], variance_.data<float>()[i], kEps);
    }
  }

  DeviceOption option_;
  std::unique_ptr<CPUContext> cpu_context_;

  TensorCPU X_;
  TensorCPU mean_;
  TensorCPU variance_;
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
    ASSERT_EQ(X_data.size(), X_.size());
    cpu_context_->Copy<float, CPUContext, CPUContext>(
        X_data.size(), X_data.data(), X_.mutable_data<float>());
    math::Transpose<float, CPUContext>(
        X_dims.size(),
        X_dims.data(),
        axes.data(),
        X_.data<float>(),
        Y_.mutable_data<float>(),
        cpu_context_.get());
    ASSERT_EQ(Y_data.size(), Y_.size());
    for (int i = 0; i < Y_.size(); ++i) {
      EXPECT_FLOAT_EQ(Y_data[i], Y_.data<float>()[i]);
    }
  }

  DeviceOption option_;
  std::unique_ptr<CPUContext> cpu_context_;

  TensorCPU X_;
  TensorCPU Y_;
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
