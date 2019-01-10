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

void AddNoiseInput(const vector<TIndex>& shape, const string& name, Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(shape);

  math::RandGaussian<float, CPUContext>(
      tensor->size(), 0.0f, 3.0f, tensor->mutable_data<float>(), &context);
  for (auto i = 0; i < tensor->size(); ++i) {
    tensor->mutable_data<float>()[i] =
        std::min(-5.0f, std::max(5.0f, tensor->mutable_data<float>()[i]));
  }
}

void compareResizeNeareast(int N,
                       int C,
                       int H,
                       int W,
                       float wscale,
                       float hscale) {
  Workspace ws;

  OperatorDef def1;
  def1.set_name("test");
  def1.set_type("ResizeNearest");
  def1.add_input("X");
  def1.add_output("Y");

  def1.add_arg()->CopyFrom(MakeArgument("width_scale", wscale));
  def1.add_arg()->CopyFrom(MakeArgument("height_scale", hscale));

  AddNoiseInput(vector<TIndex>{N, C, H, W}, "X", &ws);

  unique_ptr<OperatorBase> op1(CreateOperator(def1, &ws));
  EXPECT_NE(nullptr, op1.get());
  EXPECT_TRUE(op1->Run());

  const auto& X = ws.GetBlob("X")->Get<TensorCPU>();
  const auto& Y = ws.GetBlob("Y")->Get<TensorCPU>();

  const float* Xdata = X.data<float>();
  const float* Ydata = Y.data<float>();

  int outW = W * wscale;
  int outH = H * hscale;

  // Compare all output points
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int ph = 0; ph < outH; ++ph) {
        const int iny = std::min((int)(ph / hscale), (H - 1));
        for (int pw = 0; pw < outW; ++pw) {
          const int inx = std::min((int)(pw / wscale), (W - 1));
          const float v = Xdata[iny * W + inx];
          EXPECT_EQ(Ydata[outW * ph + pw], v);
        }
      }
      Xdata += H * W;
      Ydata += outW * outH;
    }
  }
}

int randInt(int a, int b) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  return std::uniform_int_distribution<int>(a, b)(gen);
}

TEST(ResizeNearestOp, ResizeNearest2x) {
  for (auto i = 0; i < 40; ++i) {
    auto H = randInt(1, 100);
    auto W = randInt(1, 100);
    auto C = randInt(1, 10);
    auto N = randInt(1, 2);
    compareResizeNeareast(N, C, H, W, 2.0f, 2.0f);
  }
}

} // unnamed namespace
} // namespace caffe2
