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
    const std::string& algorithm,
    const std::string& convolutionTransformStrategy,
    const std::string& activation,
    float maxRelErr,
    float absErrForRelErrFailure) {
  LOG(INFO) << "running N " << N << " inputC " << inputC << " H " << H << " W "
            << W << " outputC " << outputC << " kernelH " << kernelH
            << " kernelW " << kernelW << " strideH " << strideH << " strideW "
            << strideW << " padT " << padT << " padL " << padL << " padB "
            << padB << " padR " << padR << " group " << group;

  Workspace ws;

  OperatorDef nnpackOpDef;
  nnpackOpDef.set_name("test");
  nnpackOpDef.set_type("Conv");
  nnpackOpDef.set_engine("NNPACK");
  nnpackOpDef.add_input("X");
  nnpackOpDef.add_input("W");
  nnpackOpDef.add_input("B");
  nnpackOpDef.add_output("Y_nnpack");

  nnpackOpDef.add_arg()->CopyFrom(MakeArgument("kernel_h", kernelH));
  nnpackOpDef.add_arg()->CopyFrom(MakeArgument("kernel_w", kernelW));
  if (!algorithm.empty()) {
    nnpackOpDef.add_arg()->CopyFrom(MakeArgument("algo", algorithm));
  }
  if (!convolutionTransformStrategy.empty()) {
    nnpackOpDef.add_arg()->CopyFrom(MakeArgument(
        "convolution_transform_strategy", convolutionTransformStrategy));
  }
  if (!activation.empty()) {
    nnpackOpDef.add_arg()->CopyFrom(MakeArgument("activation", activation));
  }
  nnpackOpDef.add_arg()->CopyFrom(MakeArgument("stride_h", strideH));
  nnpackOpDef.add_arg()->CopyFrom(MakeArgument("stride_w", strideW));
  nnpackOpDef.add_arg()->CopyFrom(MakeArgument("pad_t", padT));
  nnpackOpDef.add_arg()->CopyFrom(MakeArgument("pad_l", padL));
  nnpackOpDef.add_arg()->CopyFrom(MakeArgument("pad_b", padB));
  nnpackOpDef.add_arg()->CopyFrom(MakeArgument("pad_r", padR));
  nnpackOpDef.add_arg()->CopyFrom(MakeArgument("group", group));

  AddNoiseInput(vector<int64_t>{N, inputC, H, W}, "X", &ws);
  AddNoiseInput(
      vector<int64_t>{outputC, inputC / group, kernelH, kernelW}, "W", &ws);
  AddNoiseInput(vector<int64_t>{outputC}, "B", &ws);

  unique_ptr<OperatorBase> nnpackOp(CreateOperator(nnpackOpDef, &ws));
  EXPECT_NE(nullptr, nnpackOp.get());

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

  unique_ptr<OperatorBase> activationOp;
  if (activation == "Relu") {
    OperatorDef activationOpDef;
    activationOpDef.set_name("activation");
    activationOpDef.set_type("Relu");
    activationOpDef.add_input("Y_reference");
    activationOpDef.add_output("Y_reference");
    activationOp = CreateOperator(activationOpDef, &ws);
    EXPECT_NE(nullptr, activationOp.get());
  }

  for (auto i = 0; i < 10; ++i) {
    EXPECT_TRUE(nnpackOp->Run());
  }
  Blob* nnpackOutputBlob = ws.GetBlob("Y_nnpack");
  EXPECT_NE(nullptr, nnpackOutputBlob);
  auto& nnpackOutput = nnpackOutputBlob->Get<TensorCPU>();

  for (auto i = 0; i < 10; ++i) {
    EXPECT_TRUE(referenceOp->Run());
    if (activationOp) {
      EXPECT_TRUE(activationOp->Run());
    }
  }

  Blob* referenceOutputBlob = ws.GetBlob("Y_reference");
  EXPECT_NE(nullptr, referenceOutputBlob);
  auto& referenceOutput = referenceOutputBlob->Get<TensorCPU>();

  // Compare all output points
  for (int n = 0; n < nnpackOutput.dim32(0); ++n) {
    for (int c = 0; c < nnpackOutput.dim32(1); ++c) {
      for (int h = 0; h < nnpackOutput.dim32(2); ++h) {
        for (int w = 0; w < nnpackOutput.dim32(3); ++w) {
          int offset = n * nnpackOutput.dim32(1) * nnpackOutput.dim32(2) *
                  nnpackOutput.dim32(3) +
              c * nnpackOutput.dim32(2) * nnpackOutput.dim32(3) +
              h * nnpackOutput.dim32(3) + w;

          auto v1 = nnpackOutput.data<float>()[offset];
          auto v2 = referenceOutput.data<float>()[offset];

          float relErr = relativeError(v1, v2);
          float absErr = std::abs(v1 - v2);

          // For small values / small difference, the relative error
          // can be huge but the absolute error will be small
          EXPECT_TRUE(
              relErr <= maxRelErr ||
              (relErr > maxRelErr && absErr <= absErrForRelErrFailure))
              << v1 << " " << v2 << " (rel err " << relErr << ") "
              << "(" << n << " " << c << " " << h << " " << w << ") "
              << "running N " << N << " inputC " << inputC << " H " << H
              << " W " << W << " outputC " << outputC << " kernelH " << kernelH
              << " kernelW " << kernelW << " strideH " << strideH << " strideW "
              << strideW << " padT " << padT << " padL " << padL << " padB "
              << padB << " padR " << padR << " group " << group << " algorithm "
              << algorithm << " convolutionTransformStrategy "
              << convolutionTransformStrategy;
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
    std::string algo = "",
    int planesIn = randInt(1, 6),
    int planesOut = randInt(1, 6),
    int n = randInt(1, 2),
    std::string convolutionTransformStrategy = "COMPUTE",
    std::string activation = "identity") {
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
      algo,
      convolutionTransformStrategy,
      activation,
      0.05f,
      0.1f);
}

} // unnamed namespace

constexpr int kIters = 20;

TEST(NNPACK, Conv_3x3s1) {
  for (int i = 0; i < kIters; ++i) {
    runConv(3, 3, 1, 1);
  }
}

TEST(NNPACK, Conv_3x3s1_precompute) {
  for (int i = 0; i < kIters; ++i) {
    int group = randInt(1, 2);
    runConv(
        3,
        3,
        1,
        1,
        group,
        "WINOGRAD",
        group * randInt(1, 8),
        group * randInt(1, 8),
        1,
        "PRECOMPUTE");
  }
}

TEST(NNPACK, Conv_3x3s1_FP16) {
  for (int i = 0; i < kIters; ++i) {
    runConv(3, 3, 1, 1, 1, "WINOGRAD_FP16");
  }
}

TEST(NNPACK, Conv_3x3s1_FP16_precompute) {
  for (int i = 0; i < kIters; ++i) {
    int group = randInt(1, 2);
    runConv(
        3,
        3,
        1,
        1,
        group,
        "WINOGRAD_FP16",
        group * randInt(1, 8),
        group * randInt(1, 8),
        1,
        "PRECOMPUTE");
  }
}

TEST(NNPACK, Conv_NxNs1) {
  for (int i = 0; i < kIters; ++i) {
    int kernel = randInt(2, 10);
    runConv(kernel, kernel, 1, 1);
  }
}

TEST(NNPACK, Conv_1x1s1) {
  for (int i = 0; i < kIters; ++i) {
    auto group = randInt(1, 3);
    auto inChannels = randInt(1, 8) * group;
    auto outChannels = randInt(1, 8) * group;
    auto n = 1;
    runConv(1, 1, 1, 1, group, "DIRECT", inChannels, outChannels, n);
  }
}

TEST(NNPACK, ConvRelu_1x1s1) {
  for (int i = 0; i < kIters; ++i) {
    auto group = randInt(1, 3);
    auto inChannels = randInt(1, 8) * group;
    auto outChannels = randInt(1, 8) * group;
    auto n = 1;
    runConv(
        1,
        1,
        1,
        1,
        group,
        "DIRECT",
        inChannels,
        outChannels,
        n,
        "PRECOMPUTE",
        "Relu");
  }
}

TEST(NNPACK, Conv_1x1s1_precompute) {
  for (int i = 0; i < kIters; ++i) {
    auto group = randInt(1, 3);
    auto inChannels = randInt(1, 8) * group;
    auto outChannels = randInt(1, 8) * group;
    auto n = 1;
    runConv(
        1, 1, 1, 1, group, "DIRECT", inChannels, outChannels, n, "PRECOMPUTE");
  }
}

TEST(NNPACK, Conv_NxNs_grouped) {
  for (int i = 0; i < kIters; ++i) {
    int group = randInt(2, 3);
    int iC = randInt(1, 6) * group;
    int oC = randInt(1, 6) * group;
    int kernel = randInt(2, 10);
    int n = randInt(1, 2);
    runConv(kernel, kernel, 1, 1, group, "", iC, oC, n);
  }
}

TEST(NNPACK, Conv_NxNs_grouped_precompute) {
  for (int i = 0; i < kIters; ++i) {
    int group = randInt(2, 3);
    int iC = randInt(1, 6) * group;
    int oC = randInt(1, 6) * group;
    int kernel = randInt(2, 10);
    int n = randInt(1, 2);
    runConv(kernel, kernel, 1, 1, group, "", iC, oC, n, "PRECOMPUTE");
  }
}

TEST(NNPACK, Conv_NxNsW) {
  for (int i = 0; i < 3; ++i) {
    int kernel = randInt(3, 5);
    int stride = randInt(1, kernel - 1);
    runConv(kernel, kernel, stride, stride);
  }
}

TEST(NNPACK, ConvRelu_NxNsW) {
  for (int i = 0; i < 3; ++i) {
    int kernel = randInt(3, 5);
    int stride = randInt(1, kernel - 1);
    runConv(kernel, kernel, stride, stride, 1, "", 1, 1, 1, "COMPUTE", "Relu");
  }
}

TEST(NNPACK, Conv_HxWsHxW) {
  for (int i = 0; i < 3; ++i) {
    int kernelH = randInt(2, 5);
    int kernelW = randInt(2, 5);
    int strideH = randInt(1, kernelH - 1);
    int strideW = randInt(1, kernelW - 1);
    runConv(kernelH, kernelW, strideH, strideW);
  }
}

} // namespace caffe2
