#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

#include "gtest/gtest.h"
#include <cmath>
#include <random>

namespace caffe2 {

void AddConstInput(const vector<TIndex>& shape,
                   const float value,
                   const string& name,
                   Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(shape);
  math::Set<float, CPUContext>(tensor->size(), value,
                               tensor->mutable_data<float>(),
                               &context);
}

void AddNoiseInput(const vector<TIndex>& shape,
                   const string& name,
                   Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(shape);

  math::RandGaussian<float, CPUContext>(
    tensor->size(),
    0.0f, 10.0f,
    tensor->mutable_data<float>(),
    &context);
}

inline float relativeError(float a, float b) {
  return std::abs(a - b) / (0.5f * (std::abs(a) + std::abs(b)));
}

void compare(int N, int inputC, int H, int W,
             int outputC,
             int kernelH, int kernelW, int strideH, int strideW,
             int padT, int padL, int padB, int padR,
             int adjH, int adjW,
             float maxRelErr, float absErrForRelErrFailure) {
  LOG(INFO) <<
    "running N " << N << " inputC " << inputC << " H " << H << " W " << W <<
    " outputC " << outputC <<
    " kernelH " << kernelH << " kernelW " << kernelW <<
    " strideH " << strideH << " strideW " << strideW <<
    " padT " << padT << " padL " << padL <<
    " padB " << padB << " padR " << padR <<
    " adjH " << adjH << " adjW " << adjW;

  Workspace ws;

  OperatorDef def1;
  def1.set_name("test");
  def1.set_type("ConvTranspose");
  def1.set_engine("MOBILE");
  def1.add_input("X");
  def1.add_input("W");
  def1.add_input("B");
  def1.add_output("Y1");

  def1.add_arg()->CopyFrom(MakeArgument("kernel_h", kernelH));
  def1.add_arg()->CopyFrom(MakeArgument("kernel_w", kernelW));
  def1.add_arg()->CopyFrom(MakeArgument("stride_h", strideH));
  def1.add_arg()->CopyFrom(MakeArgument("stride_w", strideW));
  def1.add_arg()->CopyFrom(MakeArgument("pad_t", padT));
  def1.add_arg()->CopyFrom(MakeArgument("pad_l", padL));
  def1.add_arg()->CopyFrom(MakeArgument("pad_b", padB));
  def1.add_arg()->CopyFrom(MakeArgument("pad_r", padR));
  def1.add_arg()->CopyFrom(MakeArgument("adj_h", adjH));
  def1.add_arg()->CopyFrom(MakeArgument("adj_w", adjW));

  AddNoiseInput(vector<TIndex>{N, inputC, H, W}, "X", &ws);
  AddNoiseInput(vector<TIndex>{inputC, outputC, kernelH, kernelW}, "W", &ws);
  AddNoiseInput(vector<TIndex>{outputC}, "B", &ws);

  unique_ptr<OperatorBase> op1(CreateOperator(def1, &ws));
  EXPECT_NE(nullptr, op1.get());

  OperatorDef def2;
  def2.set_name("test");
  def2.set_type("ConvTranspose");
  def2.add_input("X");
  def2.add_input("W");
  def2.add_input("B");
  def2.add_output("Y2");

  def2.add_arg()->CopyFrom(MakeArgument("kernel_h", kernelH));
  def2.add_arg()->CopyFrom(MakeArgument("kernel_w", kernelW));
  def2.add_arg()->CopyFrom(MakeArgument("stride_h", strideH));
  def2.add_arg()->CopyFrom(MakeArgument("stride_w", strideW));
  def2.add_arg()->CopyFrom(MakeArgument("pad_t", padT));
  def2.add_arg()->CopyFrom(MakeArgument("pad_l", padL));
  def2.add_arg()->CopyFrom(MakeArgument("pad_b", padB));
  def2.add_arg()->CopyFrom(MakeArgument("pad_r", padR));
  def2.add_arg()->CopyFrom(MakeArgument("adj_h", adjH));
  def2.add_arg()->CopyFrom(MakeArgument("adj_w", adjW));

  unique_ptr<OperatorBase> op2(CreateOperator(def2, &ws));
  EXPECT_NE(nullptr, op2.get());

  EXPECT_TRUE(op1->Run());
  Blob* Y1blob = ws.GetBlob("Y1");
  EXPECT_NE(nullptr, Y1blob);
  auto& Y1 = Y1blob->Get<TensorCPU>();

  EXPECT_TRUE(op2->Run());
  Blob* Y2blob = ws.GetBlob("Y2");
  EXPECT_NE(nullptr, Y2blob);
  auto& Y2 = Y2blob->Get<TensorCPU>();

  // Compare all output points
  for (int n = 0; n < Y1.dim32(0); ++n) {
    for (int c = 0; c < Y1.dim32(1); ++c) {
      for (int h = 0; h < Y1.dim32(2); ++h) {
        for (int w = 0; w < Y1.dim32(3); ++w) {
          int offset =
            n * Y1.dim32(1) * Y1.dim32(2) * Y1.dim32(3) +
            c * Y1.dim32(2) * Y1.dim32(3) +
            h * Y1.dim32(3) +
            w;

          auto v1 = Y1.data<float>()[offset];
          auto v2 = Y2.data<float>()[offset];

          float relErr = relativeError(v1, v2);
          float absErr = std::abs(v1 - v2);

          // For small values / small difference, the relative error
          // can be huge but the absolute error will be small
          EXPECT_TRUE(relErr <= maxRelErr ||
                      (relErr > maxRelErr &&
                       absErr <= absErrForRelErrFailure)) <<
            v1 << " " << v2 << " (rel err " << relErr << ") " <<
            "(" << n << " " << c << " " << h << " " << w << ") " <<
            "running N " << N << " inputC " << inputC <<
            " H " << H << " W " << W <<
            " outputC " << outputC <<
            " kernelH " << kernelH << " kernelW " << kernelW <<
            " strideH " << strideH << " strideW " << strideW <<
            " padT " << padT << " padL " << padL <<
            " padB " << padB << " padR " << padR <<
            " adjH " << adjH << " adjW " << adjW;

        }
      }
    }
  }
}

} // namespace caffe2

int randInt(int a, int b) {
  static std::random_device rd;
  static std::mt19937 gen(rd());

  return std::uniform_int_distribution<int>(a, b)(gen);
}

// TODO(#14383029) cblas_sgemm not yet implemented on limited mobile cases.
#if __ARM_NEON__ && !defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)
TEST(ConvTransposeMobile, Test) {
  for (int i = 0; i < 10; ++i) {
    int n = randInt(1, 3);
    int planesIn = randInt(1, 10);
    int h = randInt(10, 200);
    int w = randInt(10, 200);
    int planesOut = randInt(1, 10);
    int kernelH = randInt(2, 5);
    int kernelW = randInt(2, 5);
    int strideH = randInt(1, 4);
    int strideW = randInt(1, 4);
    int padT = randInt(0, 3);
    int padB = randInt(0, 3);
    int padL = 0;
    int padR = 0;
    int adjH = randInt(0, 3);
    if (adjH >= strideH) { adjH = strideH - 1; }
    int adjW = randInt(0, 3);
    if (adjW >= strideW) { adjW = strideW - 1; }

    caffe2::compare(n, planesIn, h, w,
                    planesOut,
                    kernelH, kernelW,
                    strideH, strideW,
                    padT, padL, padB, padR,
                    adjH, adjW, 0.002f, 0.001f);
  }
}
#endif
