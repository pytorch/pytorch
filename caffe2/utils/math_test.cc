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
  math::Set<float, CPUContext>(X.size(), 1, X.mutable_data<float>(), &cpu_context);
  math::Set<float, CPUContext>(W.size(), 1, W.mutable_data<float>(), &cpu_context);
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
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, 5, 6, 10, kOne,
                                X.data<float>(), W.data<float>(), kZero, Y.mutable_data<float>(),
                                &cpu_context);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 10) << i;
  }
  // Test Accumulate
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, 5, 6, 10, kOne,
                                X.data<float>(), W.data<float>(), kPointFive,
                                Y.mutable_data<float>(), &cpu_context);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 15) << i;
  }
  // Test Accumulate
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, 5, 6, 10,
                                kPointFive,
                                X.data<float>(), W.data<float>(), kOne, Y.mutable_data<float>(),
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
  math::Set<float, CPUContext>(X.size(), 1, X.mutable_data<float>(), &cpu_context);
  math::Set<float, CPUContext>(W.size(), 1, W.mutable_data<float>(), &cpu_context);
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
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasTrans, 5, 6, 10, kOne,
                                X.data<float>(), W.data<float>(), kZero, Y.mutable_data<float>(),
                                &cpu_context);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 10) << i;
  }
  // Test Accumulate
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasTrans, 5, 6, 10, kOne,
                                X.data<float>(), W.data<float>(), kPointFive,
                                Y.mutable_data<float>(), &cpu_context);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 15) << i;
  }
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasTrans, 5, 6, 10, kPointFive,
                                X.data<float>(), W.data<float>(), kOne, Y.mutable_data<float>(),
                                &cpu_context);
  EXPECT_EQ(Y.size(), 30);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 20) << i;
  }
}

namespace {

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
  math::Set<float, CPUContext>(A.size(), 1, A.mutable_data<float>(), &cpu_context);
  math::Set<float, CPUContext>(X.size(), 1, X.mutable_data<float>(), &cpu_context);
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
  math::Gemv<float, CPUContext>(CblasNoTrans, 5, 10, kOne, A.data<float>(), X.data<float>(),
                                kZero, Y.mutable_data<float>(), &cpu_context);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 10) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUContext>(CblasNoTrans, 5, 10, kOne, A.data<float>(), X.data<float>(),
                                kPointFive, Y.mutable_data<float>(), &cpu_context);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 15) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUContext>(CblasNoTrans, 5, 10, kPointFive, A.data<float>(),
                                X.data<float>(), kOne, Y.mutable_data<float>(),
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
  math::Set<float, CPUContext>(A.size(), 1, A.mutable_data<float>(), &cpu_context);
  math::Set<float, CPUContext>(X.size(), 1, X.mutable_data<float>(), &cpu_context);
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
  math::Gemv<float, CPUContext>(CblasTrans, 6, 10, kOne, A.data<float>(), X.data<float>(),
                                kZero, Y.mutable_data<float>(), &cpu_context);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 6) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUContext>(CblasTrans, 6, 10, kOne, A.data<float>(), X.data<float>(),
                                kPointFive, Y.mutable_data<float>(), &cpu_context);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 9) << i;
  }
  // Test Accumulate
  math::Gemv<float, CPUContext>(CblasTrans, 6, 10, kPointFive, A.data<float>(),
                                X.data<float>(), kOne, Y.mutable_data<float>(),
                                &cpu_context);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_EQ(Y.data<float>()[i], 12) << i;
  }
}

using convert::cpu_half2float;
using convert::cpu_float2half_rn;
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
    const std::vector<int> x_dims = {3};
    const std::vector<int> y_dims = {3};
    const std::vector<int> axes = {0};
    TensorCPU X(x_dims);
    TensorCPU Y(y_dims);
    for (int i = 0; i < 3; ++i) {
      X.mutable_data<float>()[i] = static_cast<float>(i + 1);
    }
    math::Transpose<float, CPUContext>(
        1,
        x_dims.data(),
        y_dims.data(),
        axes.data(),
        3,
        X.data<float>(),
        Y.mutable_data<float>(),
        &cpu_context);
    for (int i = 0; i < 3; ++i) {
      EXPECT_FLOAT_EQ(static_cast<float>(i + 1), Y.data<float>()[i]);
    }
  }

  {
    // Test for 2D transpose.
    const std::vector<int> x_dims = {2, 3};
    const std::vector<int> y_dims = {3, 2};
    const std::vector<int> axes = {1, 0};
    TensorCPU X(x_dims);
    TensorCPU Y(y_dims);
    for (int i = 0; i < 6; ++i) {
      X.mutable_data<float>()[i] = static_cast<float>(i + 1);
    }
    math::Transpose<float, CPUContext>(
        2,
        x_dims.data(),
        y_dims.data(),
        axes.data(),
        6,
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
    const std::vector<int> x_dims = {2, 2, 2};
    const std::vector<int> y_dims = {2, 2, 2};
    const std::vector<int> axes1 = {1, 2, 0};
    TensorCPU X(x_dims);
    TensorCPU Y(y_dims);
    for (int i = 0; i < 8; ++i) {
      X.mutable_data<float>()[i] = static_cast<float>(i + 1);
    }
    math::Transpose<float, CPUContext>(
        3,
        x_dims.data(),
        y_dims.data(),
        axes1.data(),
        8,
        X.data<float>(),
        Y.mutable_data<float>(),
        &cpu_context);
    const std::vector<float> expected_output1 = {
        1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f};
    for (int i = 0; i < 8; ++i) {
      EXPECT_FLOAT_EQ(expected_output1[i], Y.data<float>()[i]);
    }

    const std::vector<int> axes2 = {1, 0, 2};
    math::Set<float, CPUContext>(
        Y.size(), 0.0f, Y.mutable_data<float>(), &cpu_context);
    math::Transpose<float, CPUContext>(
        3,
        x_dims.data(),
        y_dims.data(),
        axes2.data(),
        8,
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
