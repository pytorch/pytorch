#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"
#include "gtest/gtest.h"

#include <cmath>
#include <random>

namespace caffe2 {

namespace {

void AddNoiseInput(
    const vector<int64_t>& shape,
    const string& name,
    Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = BlobGetMutableTensor(blob, CPU);
  tensor->Resize(shape);

  math::RandGaussian<float, CPUContext>(
      tensor->size(), 0.0f, 3.0f, tensor->mutable_data<float>(), &context);
  for (auto i = 0; i < tensor->size(); ++i) {
    tensor->mutable_data<float>()[i] =
        std::min(-5.0f, std::max(5.0f, tensor->mutable_data<float>()[i]));
  }
}

inline float relativeError(float a, float b) {
  return std::abs(a - b) / (0.5f * (std::abs(a) + std::abs(b)));
}

void compare(
    int N,
    int inputC,
    int H,
    int W,
    int outputC,
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int padT,
    int padL,
    int padB,
    int padR,
    int group,
    float maxRelErr,
    float absErrForRelErrFailure) {
  LOG(INFO) << "running N " << N << " inputC " << inputC << " H " << H << " W "
            << W << " outputC " << outputC << " kernelH " << kernelH
            << " kernelW " << kernelW << " strideH " << strideH << " strideW "
            << strideW << " padT " << padT << " padL " << padL << " padB "
            << padB << " padR " << padR << " group " << group;

  Workspace ws;

  OperatorDef depthwiseOpDef;
  depthwiseOpDef.set_name("test");
  depthwiseOpDef.set_type("Conv");
  depthwiseOpDef.set_engine("DEPTHWISE_3x3");
  depthwiseOpDef.add_input("X");
  depthwiseOpDef.add_input("W");
  depthwiseOpDef.add_input("B");
  depthwiseOpDef.add_output("Y_depthwise");

  depthwiseOpDef.add_arg()->CopyFrom(MakeArgument("kernel_h", kernelH));
  depthwiseOpDef.add_arg()->CopyFrom(MakeArgument("kernel_w", kernelW));
  depthwiseOpDef.add_arg()->CopyFrom(MakeArgument("stride_h", strideH));
  depthwiseOpDef.add_arg()->CopyFrom(MakeArgument("stride_w", strideW));
  depthwiseOpDef.add_arg()->CopyFrom(MakeArgument("pad_t", padT));
  depthwiseOpDef.add_arg()->CopyFrom(MakeArgument("pad_l", padL));
  depthwiseOpDef.add_arg()->CopyFrom(MakeArgument("pad_b", padB));
  depthwiseOpDef.add_arg()->CopyFrom(MakeArgument("pad_r", padR));
  depthwiseOpDef.add_arg()->CopyFrom(MakeArgument("group", group));

  AddNoiseInput(vector<int64_t>{N, inputC, H, W}, "X", &ws);
  AddNoiseInput(
      vector<int64_t>{outputC, inputC / group, kernelH, kernelW}, "W", &ws);
  AddNoiseInput(vector<int64_t>{outputC}, "B", &ws);

  unique_ptr<OperatorBase> depthwiseOp(CreateOperator(depthwiseOpDef, &ws));
  EXPECT_NE(nullptr, depthwiseOp.get());

  OperatorDef referenceOpDef;
  referenceOpDef.set_name("test");
  referenceOpDef.set_type("Conv");
  referenceOpDef.add_input("X");
  referenceOpDef.add_input("W");
  referenceOpDef.add_input("B");
  referenceOpDef.add_output("Y_reference");

  referenceOpDef.add_arg()->CopyFrom(MakeArgument("kernel_h", kernelH));
  referenceOpDef.add_arg()->CopyFrom(MakeArgument("kernel_w", kernelW));
  referenceOpDef.add_arg()->CopyFrom(MakeArgument("stride_h", strideH));
  referenceOpDef.add_arg()->CopyFrom(MakeArgument("stride_w", strideW));
  referenceOpDef.add_arg()->CopyFrom(MakeArgument("pad_t", padT));
  referenceOpDef.add_arg()->CopyFrom(MakeArgument("pad_l", padL));
  referenceOpDef.add_arg()->CopyFrom(MakeArgument("pad_b", padB));
  referenceOpDef.add_arg()->CopyFrom(MakeArgument("pad_r", padR));
  referenceOpDef.add_arg()->CopyFrom(MakeArgument("group", group));

  unique_ptr<OperatorBase> referenceOp(CreateOperator(referenceOpDef, &ws));
  EXPECT_NE(nullptr, referenceOp.get());

  for (auto i = 0; i < 10; ++i) {
    EXPECT_TRUE(depthwiseOp->Run());
  }
  Blob* depthwiseOutputBlob = ws.GetBlob("Y_depthwise");
  EXPECT_NE(nullptr, depthwiseOutputBlob);
  auto& depthwiseOutput = depthwiseOutputBlob->Get<TensorCPU>();

  for (auto i = 0; i < 10; ++i) {
    EXPECT_TRUE(referenceOp->Run());
  }

  Blob* referenceOutputBlob = ws.GetBlob("Y_reference");
  EXPECT_NE(nullptr, referenceOutputBlob);
  auto& referenceOutput = referenceOutputBlob->Get<TensorCPU>();

  // Compare all output points
  for (int n = 0; n < depthwiseOutput.dim32(0); ++n) {
    for (int c = 0; c < depthwiseOutput.dim32(1); ++c) {
      for (int h = 0; h < depthwiseOutput.dim32(2); ++h) {
        for (int w = 0; w < depthwiseOutput.dim32(3); ++w) {
          int offset = n * depthwiseOutput.dim32(1) * depthwiseOutput.dim32(2) *
                  depthwiseOutput.dim32(3) +
              c * depthwiseOutput.dim32(2) * depthwiseOutput.dim32(3) +
              h * depthwiseOutput.dim32(3) + w;

          auto v1 = depthwiseOutput.data<float>()[offset];
          auto v2 = referenceOutput.data<float>()[offset];

          float relErr = relativeError(v1, v2);
          float absErr = std::abs(v1 - v2);

          // For small values / small difference, the relative error
          // can be huge but the absolute error will be small
          EXPECT_TRUE(
              relErr <= maxRelErr || absErr <= absErrForRelErrFailure)
              << v1 << " " << v2 << " (rel err " << relErr << ") "
              << "(" << n << " " << c << " " << h << " " << w << ") "
              << "running N " << N << " inputC " << inputC << " H " << H
              << " W " << W << " outputC " << outputC << " kernelH " << kernelH
              << " kernelW " << kernelW << " strideH " << strideH << " strideW "
              << strideW << " padT " << padT << " padL " << padL << " padB "
              << padB << " padR " << padR << " group " << group;
        }
      }
    }
  }
}

int randInt(int a, int b) {
  static std::random_device rd;
  static std::mt19937 gen(rd());

  return std::uniform_int_distribution<int>(a, b)(gen);
}

void runConv(
    int kernelH,
    int kernelW,
    int strideH,
    int strideW,
    int group = 1,
    int planesIn = randInt(1, 6),
    int planesOut = randInt(1, 6),
    int n = randInt(1, 2)) {
  int h = randInt(20, 100);
  int w = randInt(20, 100);
  // This pad restriction is imposed by NNPACK
  int padT = std::min(randInt(0, 3), kernelH - 1);
  int padB = std::min(randInt(0, 3), kernelH - 1);
  int padL = std::min(randInt(0, 3), kernelW - 1);
  int padR = std::min(randInt(0, 3), kernelW - 1);

  caffe2::compare(
      n,
      planesIn,
      h,
      w,
      planesOut,
      kernelH,
      kernelW,
      strideH,
      strideW,
      padT,
      padL,
      padB,
      padR,
      group,
      0.05f,
      0.1f);
}

} // unnamed namespace

constexpr size_t kIters = 20;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(DEPTHWISE3x3, Conv) {
  for (int i = 0; i < kIters; ++i) {
    int channel = 2;
    runConv(3, 3, 1, 1, channel, channel, channel, randInt(1, 2));
  }
}

} // namespace caffe2
