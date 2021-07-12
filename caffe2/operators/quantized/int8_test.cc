#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/conv_transpose_unpool_op_base.h"
#include "caffe2/operators/spatial_batch_norm_op.h"

#include "caffe2/operators/quantized/int8_test_utils.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

// How to test

// Generate int8 tensor
// Convert to fp32
// Run with int8 backend
// Dequantize result to fp32
// Run with fp32 backend
// Compare results.

// for quantized Add, the error shouldn't exceed 2 * scale

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, ReLU) {
  auto XQ = q({1, 224, 224, 3});
  auto X = dq(*XQ);
  auto xop = CreateOperatorDef("Relu", "", {"X"}, {"Y"});
  auto op = CreateOperatorDef(
      "Int8Relu",
      "",
      {"XQ"},
      {"YQ"},
      {MakeArgument<int>("Y_zero_point", XQ->zero_point),
       MakeArgument<float>("Y_scale", XQ->scale)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_EQ(*YA, YE);
}

// LeakyReLU isn't build in xplat, so this fails buck test
// xplat/caffe2:caffe2_testAndroid
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, DISABLED_LeakyReLU) {
  auto XQ = q({1, 224, 224, 3});
  auto X = dq(*XQ);
  const float alpha = 0.1;
  auto xop = CreateOperatorDef(
      "LeakyRelu", "", {"X"}, {"Y"}, {MakeArgument<float>("alpha", alpha)});
  auto op = CreateOperatorDef(
      "Int8LeakyRelu",
      "",
      {"XQ"},
      {"YQ"},
      {MakeArgument<float>("alpha", alpha),
       MakeArgument<int>("Y_zero_point", XQ->zero_point),
       MakeArgument<float>("Y_scale", XQ->scale)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, addErrorTolerance(YQ.scale));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, Softmax) {
  auto XQ = q({1, 2, 1, 3});
  auto X = dq(*XQ);
  auto xop = CreateOperatorDef("Softmax", "", {"X"}, {"Y"});
  auto op = CreateOperatorDef(
      "Int8Softmax",
      "",
      {"XQ"},
      {"YQ"},
      {MakeArgument<int>("Y_zero_point", 0),
       MakeArgument<float>("Y_scale", 1.0 / 256)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  EXPECT_EQ(YQ.scale, 1.0 / 256);
  EXPECT_EQ(YQ.zero_point, 0);

  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, addErrorTolerance(YQ.scale));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, Sigmoid) {
  auto XQ = q({1, 2, 1, 3});
  auto X = dq(*XQ);
  auto xop = CreateOperatorDef("Sigmoid", "", {"X"}, {"Y"});
  auto op = CreateOperatorDef(
      "Int8Sigmoid",
      "",
      {"XQ"},
      {"YQ"},
      {MakeArgument<int>("Y_zero_point", 0),
       MakeArgument<float>("Y_scale", 1.0 / 256)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  EXPECT_EQ(YQ.scale, 1.0 / 256);
  EXPECT_EQ(YQ.zero_point, 0);

  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, addErrorTolerance(YQ.scale));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, MaxPool) {
  auto XQ = q({1, 25, 25, 16});
  auto X = dq(*XQ);
  auto xop = CreateOperatorDef(
      "MaxPool",
      "",
      {"X"},
      {"Y"},
      {MakeArgument<int>("kernel", 2), MakeArgument<string>("order", "NHWC")});
  auto op = CreateOperatorDef(
      "Int8MaxPool",
      "",
      {"XQ"},
      {"YQ"},
      {MakeArgument<int>("kernel", 2),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("Y_zero_point", XQ->zero_point),
       MakeArgument<float>("Y_scale", XQ->scale)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_EQ(*YA, YE);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, AveragePool) {
  auto XQ = q({1, 25, 25, 16});
  auto X = dq(*XQ);
  auto xop = CreateOperatorDef(
      "AveragePool",
      "",
      {"X"},
      {"Y"},
      {MakeArgument<int>("kernel", 2), MakeArgument<string>("order", "NHWC")});
  auto op = CreateOperatorDef(
      "Int8AveragePool",
      "",
      {"XQ"},
      {"YQ"},
      {MakeArgument<int>("kernel", 2),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("Y_zero_point", XQ->zero_point),
       MakeArgument<float>("Y_scale", XQ->scale)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, addErrorTolerance(XQ->scale));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, ResizeNearest) {
  auto XQ = q({1, 25, 25, 16});
  auto X = dq(*XQ);
  auto xop = CreateOperatorDef(
      "ResizeNearest",
      "",
      {"XT"},
      {"YT"},
      {MakeArgument<float>("width_scale", 2),
       MakeArgument<float>("height_scale", 2)});
  auto op = CreateOperatorDef(
      "Int8ResizeNearest",
      "",
      {"XQ"},
      {"YQ"},
      {MakeArgument<float>("width_scale", 2),
       MakeArgument<float>("height_scale", 2),
       MakeArgument<int>("Y_zero_point", XQ->zero_point),
       MakeArgument<float>("Y_scale", XQ->scale)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  ws.RunOperatorOnce(CreateOperatorDef("NHWC2NCHW", "", {"X"}, {"XT"}));
  ws.RunOperatorOnce(xop);
  ws.RunOperatorOnce(CreateOperatorDef("NCHW2NHWC", "", {"YT"}, {"Y"}));
  ws.RunOperatorOnce(op);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_EQ(*YA, YE);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, ChannelShuffle) {
  auto XQ = q({2, 25, 25, 32});
  auto X = dq(*XQ);
  auto xop = CreateOperatorDef(
      "ChannelShuffle",
      "",
      {"XT"},
      {"YT"},
      {
          MakeArgument<int>("kernel", 1),
          MakeArgument<int>("group", 4),
          MakeArgument<std::string>("order", "NCHW"),
      });
  auto op = CreateOperatorDef(
      "Int8ChannelShuffle",
      "",
      {"XQ"},
      {"YQ"},
      {
          MakeArgument<int>("kernel", 1),
          MakeArgument<int>("group", 4),
          MakeArgument<std::string>("order", "NHWC"),
          MakeArgument<int>("Y_zero_point", XQ->zero_point),
          MakeArgument<float>("Y_scale", XQ->scale),
      });
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  ws.RunOperatorOnce(CreateOperatorDef("NHWC2NCHW", "", {"X"}, {"XT"}));
  ws.RunOperatorOnce(xop);
  ws.RunOperatorOnce(CreateOperatorDef("NCHW2NHWC", "", {"YT"}, {"Y"}));
  ws.RunOperatorOnce(op);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_EQ(*YA, YE);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, Concat) {
  auto XQ0 = q({2, 25, 25, 16});
  auto X0 = dq(*XQ0);
  auto XQ1 = q({2, 25, 25, 24});
  auto X1 = dq(*XQ1);
  auto xop = CreateOperatorDef(
      "Concat",
      "",
      {"XT0", "XT1"},
      {"YT", "_"},
      {
          MakeArgument<std::string>("order", "NCHW"),
      });
  auto op = CreateOperatorDef(
      "Int8Concat",
      "",
      {"XQ0", "XQ1"},
      {"YQ", "_"},
      {
          MakeArgument<std::string>("order", "NHWC"),
          MakeArgument<int>("Y_zero_point", XQ0->zero_point),
          MakeArgument<float>("Y_scale", XQ0->scale),
      });
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ0")->GetMutable<int8::Int8TensorCPU>(), *XQ0);
  int8Copy(ws.CreateBlob("XQ1")->GetMutable<int8::Int8TensorCPU>(), *XQ1);
  BlobGetMutableTensor(ws.CreateBlob("X0"), CPU)->CopyFrom(*X0);
  BlobGetMutableTensor(ws.CreateBlob("X1"), CPU)->CopyFrom(*X1);
  ws.RunOperatorOnce(CreateOperatorDef("NHWC2NCHW", "", {"X0"}, {"XT0"}));
  ws.RunOperatorOnce(CreateOperatorDef("NHWC2NCHW", "", {"X1"}, {"XT1"}));
  ws.RunOperatorOnce(xop);
  ws.RunOperatorOnce(CreateOperatorDef("NCHW2NHWC", "", {"YT"}, {"Y"}));
  ws.RunOperatorOnce(op);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_EQ(*YA, YE);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, Add) {
  auto XQ0 = q({1, 10, 10, 20});
  auto XQ1 = q({1, 10, 10, 20});
  auto X0 = dq(*XQ0);
  auto X1 = dq(*XQ1);
  auto xop = CreateOperatorDef("Add", "", {"X0", "X1"}, {"Y"});
  const auto Y_scale = 2 * std::max(XQ0->scale, XQ1->scale);
  auto op = CreateOperatorDef(
      "Int8Add",
      "",
      {"XQ0", "XQ1"},
      {"YQ"},
      {MakeArgument<int>("Y_zero_point", XQ0->zero_point),
       MakeArgument<float>("Y_scale", Y_scale)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ0")->GetMutable<int8::Int8TensorCPU>(), *XQ0);
  int8Copy(ws.CreateBlob("XQ1")->GetMutable<int8::Int8TensorCPU>(), *XQ1);
  BlobGetMutableTensor(ws.CreateBlob("X0"), CPU)->CopyFrom(*X0);
  BlobGetMutableTensor(ws.CreateBlob("X1"), CPU)->CopyFrom(*X1);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, addErrorTolerance(Y_scale));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, SumRelu) {
  auto XQ0 = q({1, 10, 10, 20});
  auto XQ1 = q({1, 10, 10, 20});
  auto X0 = dq(*XQ0);
  auto X1 = dq(*XQ1);
  auto xop = CreateOperatorDef("Sum", "", {"X0", "X1"}, {"Y"});
  auto rlxop = CreateOperatorDef("Relu", "", {"Y"}, {"Y"});
  const auto Y_scale = 2 * std::max(XQ0->scale, XQ1->scale);
  auto op = CreateOperatorDef(
      "Int8SumRelu",
      "",
      {"XQ0", "XQ1"},
      {"YQ"},
      {MakeArgument<int>("Y_zero_point", XQ0->zero_point),
       MakeArgument<float>("Y_scale", Y_scale)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ0")->GetMutable<int8::Int8TensorCPU>(), *XQ0);
  int8Copy(ws.CreateBlob("XQ1")->GetMutable<int8::Int8TensorCPU>(), *XQ1);
  BlobGetMutableTensor(ws.CreateBlob("X0"), CPU)->CopyFrom(*X0);
  BlobGetMutableTensor(ws.CreateBlob("X1"), CPU)->CopyFrom(*X1);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  ws.RunOperatorOnce(rlxop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, addErrorTolerance(Y_scale));
}

void setq(int8::Int8TensorCPU* dst, const std::vector<float>& vs) {
  CHECK_EQ(vs.size(), dst->t.numel());
  for (auto i = 0; i < vs.size(); ++i) {
    uint8_t vq = std::max(
        std::numeric_limits<uint8_t>::min(),
        std::min(
            std::numeric_limits<uint8_t>::max(),
            static_cast<uint8_t>(int8::Round(
                static_cast<float>(dst->zero_point + (vs[i] / dst->scale))))));
    dst->t.mutable_data<uint8_t>()[i] = vq;
  }
}

void biassetq(int8::Int8TensorCPU* dst, const std::vector<float>& vs) {
  CHECK_EQ(vs.size(), dst->t.numel());
  for (auto i = 0; i < vs.size(); ++i) {
    int32_t vq = std::max(
        std::numeric_limits<int32_t>::min(),
        std::min(
            std::numeric_limits<int32_t>::max(),
            static_cast<int32_t>(int8::Round(
                static_cast<float>(dst->zero_point + (vs[i] / dst->scale))))));
    dst->t.mutable_data<int32_t>()[i] = vq;
  }
}

// Use TFLite test vectors to ensure compatibility.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, Conv) {
  auto XQ = q({2, 2, 4, 1});
  XQ->scale = 0.5;
  XQ->zero_point = 127;
  setq(
      XQ.get(),
      std::vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 3, 4, 1, 2, 3, 4});
  auto WQ = q({3, 2, 2, 1});
  WQ->scale = 0.5;
  WQ->zero_point = 127;
  setq(
      WQ.get(),
      {
          1,
          2,
          3,
          4,
          -1,
          1,
          -1,
          1,
          -1,
          -1,
          1,
          1,
      });
  auto BQ = biasq({3}, XQ->scale * WQ->scale);
  biassetq(BQ.get(), {1, 2, 3});
  auto X = dq(*XQ);
  auto W = dq(*WQ);
  auto B = biasdq(*BQ);
  auto xop = CreateOperatorDef(
      "Conv",
      "",
      {"X", "W", "B"},
      {"Y"},
      {MakeArgument<int>("kernel", 2),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("stride", 2)});
  auto op = CreateOperatorDef(
      "Int8Conv",
      "",
      {"XQ", "WQ", "BQ"},
      {"YQ"},
      {MakeArgument<int>("kernel", 2),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("stride", 2),
       MakeArgument<int>("Y_zero_point", 127),
       MakeArgument<float>("Y_scale", 1.0)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  int8Copy(ws.CreateBlob("WQ")->GetMutable<int8::Int8TensorCPU>(), *WQ);
  int8Copy(ws.CreateBlob("BQ")->GetMutable<int8::Int8TensorCPU>(), *BQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  BlobGetMutableTensor(ws.CreateBlob("W"), CPU)->CopyFrom(*W);
  BlobGetMutableTensor(ws.CreateBlob("B"), CPU)->CopyFrom(*B);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TRUE(
      (std::vector<uint8_t>(
           YQ.t.data<uint8_t>(), YQ.t.data<uint8_t>() + YQ.t.numel()) ==
       std::vector<uint8_t>{
           145, 129, 132, 145, 129, 132, 144, 131, 130, 164, 131, 130}));

  EXPECT_TENSOR_APPROX_EQ(*YA, YE, 1.0e-5);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, Grouped1x1Conv) {
  auto XQ = q({1, 3, 2, 4});
  XQ->scale = 0.5;
  XQ->zero_point = 127;
  setq(XQ.get(), std::vector<float>{1, 4, 3, 2, 9, 3, 8, 2, 6, 7, 8, 2,
                                    3, 8, 1, 7, 4, 2, 1, 3, 8, 5, 3, 1});

  // G = 2
  auto WQ = q({4, 1, 1, 2});
  WQ->scale = 0.5;
  WQ->zero_point = 127;
  setq(WQ.get(), {1, 2, 3, 4, -1, -2, -3, -4});
  auto BQ = biasq({4}, XQ->scale * WQ->scale);
  biassetq(BQ.get(), {1, 2, 3, 4});
  auto X = dq(*XQ);
  auto W = dq(*WQ);
  auto B = biasdq(*BQ);
  auto xop = CreateOperatorDef(
      "Conv",
      "",
      {"XT", "WT", "B"},
      {"YT"},
      {MakeArgument<int>("kernel", 1),
       MakeArgument<string>("order", "NCHW"),
       MakeArgument<int>("group", 2)});
  auto op = CreateOperatorDef(
      "Int8Conv",
      "",
      {"XQ", "WQ", "BQ"},
      {"YQ"},
      {MakeArgument<int>("kernel", 1),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("group", 2),
       MakeArgument<int>("Y_zero_point", 127),
       MakeArgument<float>("Y_scale", 1.0)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  int8Copy(ws.CreateBlob("WQ")->GetMutable<int8::Int8TensorCPU>(), *WQ);
  int8Copy(ws.CreateBlob("BQ")->GetMutable<int8::Int8TensorCPU>(), *BQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  BlobGetMutableTensor(ws.CreateBlob("W"), CPU)->CopyFrom(*W);
  BlobGetMutableTensor(ws.CreateBlob("B"), CPU)->CopyFrom(*B);
  ws.RunOperatorOnce(op);

  ws.RunOperatorOnce(CreateOperatorDef("NHWC2NCHW", "", {"X"}, {"XT"}));
  // Need to transpose MxKHxKWx1 to Mx1xKHxKW
  ws.RunOperatorOnce(CreateOperatorDef("NHWC2NCHW", "", {"W"}, {"WT"}));
  ws.RunOperatorOnce(xop);
  ws.RunOperatorOnce(CreateOperatorDef("NCHW2NHWC", "", {"YT"}, {"Y"}));
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, 1.0e-5);

  // test repacking between runs
  std::unique_ptr<OperatorBase> op_ptr(CreateOperator(op, &ws));
  EXPECT_TRUE(op_ptr != nullptr);
  for (auto it = 0; it < 3; ++it) {
    EXPECT_TRUE(op_ptr->Run());
    const auto& temp_YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
    auto temp_YA = dq(temp_YQ);
    const auto& temp_YE = ws.GetBlob("Y")->Get<TensorCPU>();
    EXPECT_TENSOR_APPROX_EQ(*temp_YA, temp_YE, 1.0e-5);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, Conv2) {
  auto XQ = q({1, 3, 6, 1});
  XQ->scale = 0.5;
  XQ->zero_point = 127;
  setq(
      XQ.get(),
      std::vector<float>{
          3, 2, 1, -1, -2, -3, 4, 3, 2, -2, -3, -4, 5, 4, 3, -3, -4, -5});
  auto WQ = q({1, 2, 2, 1});
  WQ->scale = 0.5;
  WQ->zero_point = 127;
  setq(WQ.get(), {1, 2, 3, 4});
  auto BQ = biasq({1}, XQ->scale * WQ->scale);
  biassetq(BQ.get(), {-1});
  auto X = dq(*XQ);
  auto W = dq(*WQ);
  auto B = biasdq(*BQ);
  auto xop = CreateOperatorDef(
      "Conv",
      "",
      {"X", "W", "B"},
      {"Y"},
      {MakeArgument<int>("kernel", 2),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("stride_w", 3),
       MakeArgument<int>("stride_h", 1)});
  auto op = CreateOperatorDef(
      "Int8Conv",
      "",
      {"XQ", "WQ", "BQ"},
      {"YQ"},
      {MakeArgument<int>("kernel", 2),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("stride_w", 3),
       MakeArgument<int>("stride_h", 1),
       MakeArgument<int>("Y_zero_point", 127),
       MakeArgument<float>("Y_scale", 1.0)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  int8Copy(ws.CreateBlob("WQ")->GetMutable<int8::Int8TensorCPU>(), *WQ);
  int8Copy(ws.CreateBlob("BQ")->GetMutable<int8::Int8TensorCPU>(), *BQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  BlobGetMutableTensor(ws.CreateBlob("W"), CPU)->CopyFrom(*W);
  BlobGetMutableTensor(ws.CreateBlob("B"), CPU)->CopyFrom(*B);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TRUE(
      (std::vector<uint8_t>(
           YQ.t.data<uint8_t>(), YQ.t.data<uint8_t>() + YQ.t.numel()) ==
       std::vector<uint8_t>{157, 103, 167, 93}));
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, 1.0e-5);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, DepthwiseConv) {
  auto XQ = q({1, 3, 2, 2});
  XQ->scale = 0.5;
  XQ->zero_point = 127;
  // setq(XQ.get(), std::vector<float>{1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12});
  setq(XQ.get(), std::vector<float>{1, 4, 3, 2, 9, 3, 8, 2, 6, 7, 8, 2});

  auto WQ = q({2, 2, 2, 1});
  WQ->scale = 0.5;
  WQ->zero_point = 127;
  setq(WQ.get(), {1, 2, 3, 4, -9, 10, -11, 12});
  auto BQ = biasq({2}, XQ->scale * WQ->scale);
  biassetq(BQ.get(), {1, 2});
  auto X = dq(*XQ);
  auto W = dq(*WQ);
  auto B = biasdq(*BQ);
  auto xop = CreateOperatorDef(
      "Conv",
      "",
      {"XT", "WT", "B"},
      {"YT"},
      {MakeArgument<int>("kernel", 2),
       MakeArgument<string>("order", "NCHW"),
       MakeArgument<int>("group", 2)});
  auto op = CreateOperatorDef(
      "Int8Conv",
      "",
      {"XQ", "WQ", "BQ"},
      {"YQ"},
      {MakeArgument<int>("kernel", 2),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("group", 2),
       MakeArgument<int>("Y_zero_point", 127),
       MakeArgument<float>("Y_scale", 1.0)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  int8Copy(ws.CreateBlob("WQ")->GetMutable<int8::Int8TensorCPU>(), *WQ);
  int8Copy(ws.CreateBlob("BQ")->GetMutable<int8::Int8TensorCPU>(), *BQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  BlobGetMutableTensor(ws.CreateBlob("W"), CPU)->CopyFrom(*W);
  BlobGetMutableTensor(ws.CreateBlob("B"), CPU)->CopyFrom(*B);
  ws.RunOperatorOnce(op);

  ws.RunOperatorOnce(CreateOperatorDef("NHWC2NCHW", "", {"X"}, {"XT"}));
  // Need to transpose MxKHxKWx1 to Mx1xKHxKW
  ws.RunOperatorOnce(CreateOperatorDef("NHWC2NCHW", "", {"W"}, {"WT"}));
  ws.RunOperatorOnce(xop);
  ws.RunOperatorOnce(CreateOperatorDef("NCHW2NHWC", "", {"YT"}, {"Y"}));
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  for (auto i = 0; i < YA->numel(); ++i) {
    LOG(INFO) << YA->data<float>()[i];
    LOG(INFO) << YE.data<float>()[i];
  }
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, 1.0e-5);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, DepthwiseConv3x3) {
  auto XQ = q({1, 3, 3, 3});
  XQ->scale = 0.5;
  XQ->zero_point = 127;
  setq(XQ.get(), std::vector<float>{1, 4, 3, 2, 9, 3, 8, 2, 6,
                                    7, 8, 2, 3, 4, 5, 2, 4, 4,
                                    9, 8, 7, 6, 5, 4, 3, 2, 1});

  auto WQ = q({3, 3, 3, 1});
  WQ->scale = 0.5;
  WQ->zero_point = 127;
  setq(WQ.get(), std::vector<float>{1, -4, 3, 2, -9, 3, -8, 2, 6,
                                    7, 8, -2, -3, 4, -5, -2, 4, 4,
                                    -9, 8, -7, 6, -5, 4, 3, -2, 1});
  auto BQ = biasq({3}, XQ->scale * WQ->scale);
  biassetq(BQ.get(), {1, 2, 3});
  auto X = dq(*XQ);
  auto W = dq(*WQ);
  auto B = biasdq(*BQ);
  auto xop = CreateOperatorDef(
      "Conv",
      "",
      {"XT", "WT", "B"},
      {"YT"},
      {MakeArgument<int>("kernel", 3),
       MakeArgument<string>("order", "NCHW"),
       MakeArgument<int>("group", 3)});
  auto op = CreateOperatorDef(
      "Int8Conv",
      "",
      {"XQ", "WQ", "BQ"},
      {"YQ"},
      {MakeArgument<int>("kernel", 3),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("group", 3),
       MakeArgument<int>("Y_zero_point", 127),
       MakeArgument<float>("Y_scale", 1.0)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  int8Copy(ws.CreateBlob("WQ")->GetMutable<int8::Int8TensorCPU>(), *WQ);
  int8Copy(ws.CreateBlob("BQ")->GetMutable<int8::Int8TensorCPU>(), *BQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  BlobGetMutableTensor(ws.CreateBlob("W"), CPU)->CopyFrom(*W);
  BlobGetMutableTensor(ws.CreateBlob("B"), CPU)->CopyFrom(*B);
  ws.RunOperatorOnce(op);

  ws.RunOperatorOnce(CreateOperatorDef("NHWC2NCHW", "", {"X"}, {"XT"}));
  // Need to transpose MxKHxKWx1 to Mx1xKHxKW
  ws.RunOperatorOnce(CreateOperatorDef("NHWC2NCHW", "", {"W"}, {"WT"}));
  ws.RunOperatorOnce(xop);
  ws.RunOperatorOnce(CreateOperatorDef("NCHW2NHWC", "", {"YT"}, {"Y"}));
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  for (auto i = 0; i < YA->numel(); ++i) {
    LOG(INFO) << YA->data<float>()[i];
    LOG(INFO) << YE.data<float>()[i];
  }
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, 1.0e-5);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, DepthwiseConv5x5) {
  auto XQ = q({1, 5, 5, 1});
  XQ->scale = 0.5;
  XQ->zero_point = 127;
  setq(XQ.get(), std::vector<float>{1, 4, 3, 2, 9, 3, 8, 2, 6,
                                    7, 8, 2, 3, 4, 5, 2, 4, 4,
                                    9, 8, 7, 6, 5, 4, 3});

  auto WQ = q({1, 5, 5, 1});
  WQ->scale = 0.5;
  WQ->zero_point = 127;
  setq(WQ.get(), std::vector<float>{1, -4, 3, 2, -9, 3, -8, 2, 6,
                                    7, 8, -2, -3, 4, -5, -2, 4, 4,
                                    -9, 8, -7, 6, -5, 4, 3});
  auto BQ = biasq({1}, XQ->scale * WQ->scale);
  biassetq(BQ.get(), {1});
  auto X = dq(*XQ);
  auto W = dq(*WQ);
  auto B = biasdq(*BQ);
  auto xop = CreateOperatorDef(
      "Conv",
      "",
      {"XT", "WT", "B"},
      {"YT"},
      {MakeArgument<int>("kernel", 5),
       MakeArgument<string>("order", "NCHW"),
       MakeArgument<int>("group", 1)});
  auto op = CreateOperatorDef(
      "Int8Conv",
      "",
      {"XQ", "WQ", "BQ"},
      {"YQ"},
      {MakeArgument<int>("kernel", 5),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("group", 1),
       MakeArgument<int>("Y_zero_point", 127),
       MakeArgument<float>("Y_scale", 1.0)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  int8Copy(ws.CreateBlob("WQ")->GetMutable<int8::Int8TensorCPU>(), *WQ);
  int8Copy(ws.CreateBlob("BQ")->GetMutable<int8::Int8TensorCPU>(), *BQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  BlobGetMutableTensor(ws.CreateBlob("W"), CPU)->CopyFrom(*W);
  BlobGetMutableTensor(ws.CreateBlob("B"), CPU)->CopyFrom(*B);
  ws.RunOperatorOnce(op);

  ws.RunOperatorOnce(CreateOperatorDef("NHWC2NCHW", "", {"X"}, {"XT"}));
  // Need to transpose MxKHxKWx1 to Mx1xKHxKW
  ws.RunOperatorOnce(CreateOperatorDef("NHWC2NCHW", "", {"W"}, {"WT"}));
  ws.RunOperatorOnce(xop);
  ws.RunOperatorOnce(CreateOperatorDef("NCHW2NHWC", "", {"YT"}, {"Y"}));
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  for (auto i = 0; i < YA->numel(); ++i) {
    LOG(INFO) << YA->data<float>()[i];
    LOG(INFO) << YE.data<float>()[i];
  }
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, 1.0e-5);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, ConvTranspose) {
  auto XQ = q({1, 3, 6, 1});
  XQ->scale = 0.5;
  XQ->zero_point = 127;
  setq(
      XQ.get(),
      std::vector<float>{
          3, 2, 1, -1, -2, -3, 4, 3, 2, -2, -3, -4, 5, 4, 3, -3, -4, -5});
  auto WQ = q({1, 2, 2, 1});
  WQ->scale = 0.5;
  WQ->zero_point = 127;
  setq(WQ.get(), {1, 2, 3, 4});
  auto BQ = biasq({1}, XQ->scale * WQ->scale);
  biassetq(BQ.get(), {-1});
  auto X = dq(*XQ);
  auto W = dq(*WQ);
  auto B = biasdq(*BQ);
  auto xop = CreateOperatorDef(
      "ConvTranspose",
      "",
      {"X", "W", "B"},
      {"Y"},
      {MakeArgument<int>("kernel", 2),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("stride_w", 1),
       MakeArgument<int>("stride_h", 2)});
  auto op = CreateOperatorDef(
      "Int8ConvTranspose",
      "",
      {"XQ", "WQ", "BQ"},
      {"YQ"},
      {MakeArgument<int>("kernel", 2),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("stride_w", 1),
       MakeArgument<int>("stride_h", 2),
       MakeArgument<int>("Y_zero_point", 127),
       MakeArgument<float>("Y_scale", 1.0)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  int8Copy(ws.CreateBlob("WQ")->GetMutable<int8::Int8TensorCPU>(), *WQ);
  int8Copy(ws.CreateBlob("BQ")->GetMutable<int8::Int8TensorCPU>(), *BQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  BlobGetMutableTensor(ws.CreateBlob("W"), CPU)->CopyFrom(*W);
  BlobGetMutableTensor(ws.CreateBlob("B"), CPU)->CopyFrom(*B);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, 1.0e-5);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, FC) {
  auto XQ = q({2, 10});
  XQ->scale = 0.5;
  XQ->zero_point = 127;
  setq(XQ.get(), {1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,
                  1, 2, 3, 4, 5, 6, 7, -8, 9,  -10});
  auto WQ = q({3, 10});
  WQ->scale = 0.5;
  WQ->zero_point = 127;
  setq(
      WQ.get(),
      {
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // u = 0
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // u = 1
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // u = 1
      });
  auto BQ = biasq({3}, XQ->scale * WQ->scale);
  biassetq(BQ.get(), {1, 2, 3});
  auto X = dq(*XQ);
  auto W = dq(*WQ);
  auto B = biasdq(*BQ);
  auto xop = CreateOperatorDef("FC", "", {"X", "W", "B"}, {"Y"}, {});
  auto op = CreateOperatorDef(
      "Int8FC",
      "",
      {"XQ", "WQ", "BQ"},
      {"YQ"},
      {MakeArgument<int>("Y_zero_point", 127),
       MakeArgument<float>("Y_scale", 1.0)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  int8Copy(ws.CreateBlob("WQ")->GetMutable<int8::Int8TensorCPU>(), *WQ);
  int8Copy(ws.CreateBlob("BQ")->GetMutable<int8::Int8TensorCPU>(), *BQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  BlobGetMutableTensor(ws.CreateBlob("W"), CPU)->CopyFrom(*W);
  BlobGetMutableTensor(ws.CreateBlob("B"), CPU)->CopyFrom(*B);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  for (auto i = 0; i < YA->numel(); ++i) {
    LOG(INFO) << YA->data<float>()[i];
    LOG(INFO) << YE.data<float>()[i];
  }
  EXPECT_TRUE(
      (std::vector<uint8_t>(
           YQ.t.data<uint8_t>(), YQ.t.data<uint8_t>() + YQ.t.numel()) ==
       std::vector<uint8_t>{151, 152, 153, 185, 186, 187}));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, GivenTensorFill) {
  vector<int64_t> shape = {1, 25, 25, 16};
  auto XQ = q(shape);
  auto X = dq(*XQ);
  vector<float> v(
      X->template data<float>(), X->template data<float>() + X->numel());
  std::string vq(
      XQ->t.template data<uint8_t>(),
      XQ->t.template data<uint8_t>() + XQ->t.numel());
  auto op = CreateOperatorDef(
      "GivenTensorFill",
      "",
      {},
      {"Y"},
      {MakeArgument<vector<int64_t>>("shape", shape),
       MakeArgument<vector<float>>("values", v)});
  auto xop = CreateOperatorDef(
      "Int8GivenTensorFill",
      "",
      {},
      {"YQ"},
      {MakeArgument<vector<int64_t>>("shape", shape),
       MakeArgument<string>("values", vq),
       MakeArgument<float>("Y_scale", XQ->scale),
       MakeArgument<int32_t>("Y_zero_point", XQ->zero_point)});
  Workspace ws;
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, 1.0e-5);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, GivenIntTensorFill) {
  vector<int64_t> shape = {32};
  auto XQ = biasq(shape, 1. / 255 * 1. / 255);
  auto X = biasdq(*XQ);
  vector<float> v(
      X->template data<float>(), X->template data<float>() + X->numel());
  vector<int32_t> vq(
      XQ->t.template data<int32_t>(),
      XQ->t.template data<int32_t>() + XQ->t.numel());
  auto op = CreateOperatorDef(
      "GivenTensorFill",
      "",
      {},
      {"Y"},
      {MakeArgument<vector<int64_t>>("shape", shape),
       MakeArgument<vector<float>>("values", v)});
  auto xop = CreateOperatorDef(
      "Int8GivenIntTensorFill",
      "",
      {},
      {"YQ"},
      {MakeArgument<vector<int64_t>>("shape", shape),
       MakeArgument<vector<int32_t>>("values", vq),
       MakeArgument<float>("Y_scale", XQ->scale),
       MakeArgument<int32_t>("Y_zero_point", XQ->zero_point)});
  Workspace ws;
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = biasdq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, 1.0e-5);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, QuantDeQuant) {
  vector<int64_t> shape = {1, 25, 25, 16};
  auto XQ = q(shape);
  auto X = dq(*XQ);
  auto xop = CreateOperatorDef(
      "Int8Quantize",
      "",
      {"X"},
      {"XQ"},
      {MakeArgument<float>("Y_scale", XQ->scale),
       MakeArgument<int32_t>("Y_zero_point", XQ->zero_point)});
  auto op = CreateOperatorDef("Int8Dequantize", "", {"XQ"}, {"X_x"});
  Workspace ws;
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  ws.RunOperatorOnce(xop);
  ws.RunOperatorOnce(op);
  const auto& X_x = ws.GetBlob("X_x")->Get<TensorCPU>();
  EXPECT_TENSOR_APPROX_EQ(*X, X_x, XQ->scale);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, Reshape) {
  auto XQ = q({1, 25, 25, 16});
  auto xop = CreateOperatorDef(
      "Int8Reshape",
      "",
      {"XQ"},
      {"YQ", "old_shape"},
      {MakeArgument("shape", vector<int64_t>{0, -1, 2000}),
       MakeArgument<float>("Y_scale", XQ->scale),
       MakeArgument<int32_t>("Y_zero_point", XQ->zero_point)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  EXPECT_EQ(YQ.t.sizes(), (vector<int64_t>{1, 5, 2000}));
  EXPECT_EQ(YQ.scale, XQ->scale);
  EXPECT_EQ(YQ.zero_point, XQ->zero_point);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, Flatten) {
  auto XQ = q({1, 25, 25, 16});
  auto xop = CreateOperatorDef(
      "Int8Flatten",
      "",
      {"XQ"},
      {"YQ"},
      {MakeArgument<int>("axis", 2),
       MakeArgument<float>("Y_scale", XQ->scale),
       MakeArgument<int32_t>("Y_zero_point", XQ->zero_point)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  EXPECT_EQ(YQ.t.sizes(), (vector<int64_t>{25, 400}));
  EXPECT_EQ(YQ.scale, XQ->scale);
  EXPECT_EQ(YQ.zero_point, XQ->zero_point);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, Slice) {
  auto XQ = q({1, 25, 25, 16});
  auto X = dq(*XQ);
  vector<int> starts = {0, 3, 0, 0};
  vector<int> ends = {-1, 5, -1, -1};
  auto xop = CreateOperatorDef(
      "Slice",
      "",
      {"X"},
      {"Y"},
      {MakeArgument<vector<int>>("starts", starts),
       MakeArgument<vector<int>>("ends", ends)});
  auto op = CreateOperatorDef(
      "Int8Slice",
      "",
      {"XQ"},
      {"YQ"},
      {MakeArgument<vector<int>>("starts", starts),
       MakeArgument<vector<int>>("ends", ends),
       MakeArgument<int>("Y_zero_point", XQ->zero_point),
       MakeArgument<float>("Y_scale", XQ->scale)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  EXPECT_TENSOR_EQ(*YA, YE);
  EXPECT_EQ(YQ.t.sizes(), (vector<int64_t>{1, 2, 25, 16}));
  EXPECT_EQ(YQ.scale, XQ->scale);
  EXPECT_EQ(YQ.zero_point, XQ->zero_point);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(Int8, DISABLED_Transpose) {
  auto XQ = q({1, 50, 25, 16});
  auto xop = CreateOperatorDef(
      "Int8Transpose",
      "",
      {"XQ"},
      {"YQ"},
      {MakeArgument("axes", vector<int64_t>{0, 3, 1, 2}),
       MakeArgument<float>("Y_scale", XQ->scale),
       MakeArgument<int32_t>("Y_zero_point", XQ->zero_point)});
  Workspace ws;
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  EXPECT_EQ(YQ.t.sizes(), (vector<int64_t>{1, 16, 50, 25}));
  EXPECT_EQ(YQ.scale, XQ->scale);
  EXPECT_EQ(YQ.zero_point, XQ->zero_point);
}
} // namespace caffe2
