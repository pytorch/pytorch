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

void AddNoiseInput(const vector<int64_t>& shape, const string& name, Workspace* ws) {
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

void compareMaxPooling(int N,
                       int C,
                       int H,
                       int W,
                       int kernelH,
                       int kernelW,
                       int strideH,
                       int strideW,
                       int padT,
                       int padL,
                       int padB,
                       int padR,
                       float maxRelErr = 1.0e-5f,
                       float absErrForRelErrFailure = 1.0e-5f) {
  Workspace ws;

  OperatorDef def1;
  def1.set_name("test");
  def1.set_type("MaxPool");
  def1.add_input("X");
  def1.add_output("Y");

  def1.add_arg()->CopyFrom(MakeArgument("kernel_h", kernelH));
  def1.add_arg()->CopyFrom(MakeArgument("kernel_w", kernelW));
  def1.add_arg()->CopyFrom(MakeArgument("stride_h", strideH));
  def1.add_arg()->CopyFrom(MakeArgument("stride_w", strideW));
  def1.add_arg()->CopyFrom(MakeArgument("pad_t", padT));
  def1.add_arg()->CopyFrom(MakeArgument("pad_l", padL));
  def1.add_arg()->CopyFrom(MakeArgument("pad_b", padB));
  def1.add_arg()->CopyFrom(MakeArgument("pad_r", padR));

  AddNoiseInput(vector<int64_t>{N, C, H, W}, "X", &ws);

  unique_ptr<OperatorBase> op1(CreateOperator(def1, &ws));
  EXPECT_NE(nullptr, op1.get());
  EXPECT_TRUE(op1->Run());

  const auto& X = ws.GetBlob("X")->Get<TensorCPU>();
  const auto& Y = ws.GetBlob("Y")->Get<TensorCPU>();

  // Compare all output points
  for (int n = 0; n < Y.dim32(0); ++n) {
    for (int c = 0; c < Y.dim32(1); ++c) {
      for (int ph = 0; ph < Y.dim32(2); ++ph) {
        for (int pw = 0; pw < Y.dim32(3); ++pw) {
          // Reference implementations
          int hstart = ph * strideH - padT;
          int wstart = pw * strideW - padL;
          int hend = std::min(hstart + kernelH, H);
          int wend = std::min(wstart + kernelW, W);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          const int pool_index = ph * Y.dim32(3) + pw;
          float v = std::numeric_limits<float>::lowest();
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const auto* Xdata =
                  X.data<float>() + n * X.dim(1) * X.dim(2) * X.dim(3) + c * X.dim(2) * X.dim(3);
              const int input_index = h * W + w;
              v = std::max(v, Xdata[input_index]);
            }
          }
          EXPECT_EQ(Y.data<float>()[n * Y.dim(1) * Y.dim(2) * Y.dim(3) + c * Y.dim(2) * Y.dim(3) +
                                    ph * Y.dim(3) + pw],
                    v);
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

void runMaxPool(int kernel, int stride, int pad) {
  int N = randInt(1, 2);
  int C = randInt(1, 12);
  int H = randInt(50, 100);
  int W = randInt(50, 100);
  int planesOut = randInt(1, 6);

  compareMaxPooling(N, C, H, W, kernel, kernel, stride, stride, pad, pad, pad, pad);
}

TEST(PoolOp, MaxPool2x2s2p0Randomized) {
  for (int i = 0; i < 40; ++i) {
    runMaxPool(2, 2, 0);
  }
}

TEST(PoolOp, MaxPool4x4s3p2Randomized) {
  for (int i = 0; i < 40; ++i) {
    runMaxPool(4, 3, 2);
  }
}

TEST(PoolOp, MaxPool2x2s2p0Special) {
  // 2x2s2p0 where H/W % 4 == 0
  compareMaxPooling(2, 10, 40, 40, 2, 2, 2, 2, 0, 0, 0, 0, 0.05f, 0.1f);

  // 2x2s2p0 where H/W % 4 != 0
  compareMaxPooling(2, 10, 39, 39, 2, 2, 2, 2, 0, 0, 0, 0, 0.05f, 0.1f);

  // 2x2s2p0 where H/W % 16 == 0
  compareMaxPooling(2, 10, 64, 64, 2, 2, 2, 2, 0, 0, 0, 0, 0.05f, 0.1f);
}

TEST(PoolOp, MaxPoolFullyRandomized) {
  for (auto i = 0; i < 40; ++i) {
    auto kernelH = randInt(1, 5);
    auto kernelW = randInt(1, 5);
    auto strideH = randInt(1, 5);
    auto strideW = randInt(1, 5);
    auto padL = randInt(0, kernelW - 1);
    auto padR = randInt(0, kernelW - 1);
    auto padT = randInt(0, kernelH - 1);
    auto padB = randInt(0, kernelH - 1);
    auto H = randInt(std::max(1, kernelH - padT - padB), 100);
    auto W = randInt(std::max(1, kernelW - padL - padR), 100);
    auto C = randInt(1, 10);
    auto N = randInt(1, 2);
    compareMaxPooling(
        N, C, H, W, kernelH, kernelW, strideH, strideW, padT, padL, padB, padR);
  }
}
} // unnamed namespace


} // namespace caffe2
