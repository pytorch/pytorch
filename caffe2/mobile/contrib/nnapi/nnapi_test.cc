#include <gtest/gtest.h>

#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

#include "NeuralNetworks.h"
#include "nnapi.h"

namespace caffe2 {

namespace {

// CPU: t1, NN-API: t2
void checkError(const TensorCPU& t1, const TensorCPU& t2, float error) {
  CAFFE_ENFORCE_EQ(
      t1.dims(),
      t2.dims(),
      "t1.size() = ",
      t1.size(),
      ", t2.size() = ",
      t2.size());
  float maxError = 0, minError = 0;
  if (t1.template IsType<float>()) {
    for (auto i = 0; i < t1.size(); ++i) {
      const float t1_i = t1.template data<float>()[i];
      const float t2_i = t2.template data<float>()[i];
      EXPECT_NEAR(t1_i, t2_i, error);
      float err = t1_i - t2_i;
      if (err > maxError) {
        maxError = err;
      } else if (err < minError) {
        minError = err;
      }
    }
  } else if (t1.template IsType<uint8_t>()) {
    for (auto i = 0; i < t1.size(); ++i) {
      const uint8_t t1_i = t1.template data<uint8_t>()[i];
      const uint8_t t2_i = t2.template data<uint8_t>()[i];
      EXPECT_NEAR(t1_i, t2_i, error);
      float err = t1_i - t2_i;
      if (err > maxError) {
        maxError = err;
      } else if (err < minError) {
        minError = err;
      }
    }
  }
  LOG(ERROR) << "maxError = " << maxError << ", minError = " << minError;
}

static void test_relu(int N, int C, int H, int W) {
  // CPU reference
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, H, W, C);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }
  NetDef netdef;
  {
    {
      auto& op = *(netdef.add_op());
      op.set_type("Relu");
      op.add_input("X_cpu");
      op.add_output("Y_cpu");
    }
    netdef.add_external_input("X_cpu");
    netdef.add_external_output("Y_cpu");
  }
  ws.RunNetOnce(netdef);
  const auto& t_cpu = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // cpu

  // NN API
  netdef.mutable_op(0)->set_output(0, "Y_nn");
  netdef.set_external_output(0, "Y_nn");
  NetDef initNet;
  NNApi model(initNet, netdef, &ws);
  std::vector<TensorCPU*> inputs, outputs;
  inputs.push_back(ws.GetBlob("X_cpu")->GetMutable<TensorCPU>());
  EXPECT_TRUE(model.run(inputs, &outputs));
  const auto& t_nn = *outputs[0];

  checkError(t_cpu, t_nn, 0.01);
}

static void test_conv_NHWC(
    int N,
    int C,
    int H,
    int W,
    int K,
    int kernel,
    int pad_t,
    int pad_l,
    int pad_b,
    int pad_r,
    int stride_h,
    int stride_w) {
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, H, W, C);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }
  {
    auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
    t->Resize(K, kernel, kernel, C);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }
  {
    auto* t = ws.CreateBlob("B")->GetMutable<TensorCPU>();
    t->Resize(K);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    {
      auto& op = *(netdef.add_op());
      op.set_type("Conv");
      op.add_input("X_cpu");
      op.add_input("W");
      op.add_input("B");
      op.add_output("Y_cpu");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("order");
        arg.set_s("NHWC");
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("kernel");
        arg.set_i(kernel);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_t");
        arg.set_i(pad_t);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_l");
        arg.set_i(pad_l);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_b");
        arg.set_i(pad_b);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_r");
        arg.set_i(pad_r);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("stride_h");
        arg.set_i(stride_h);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("stride_w");
        arg.set_i(stride_w);
      }
    }
    netdef.add_external_input("X_cpu");
    netdef.add_external_input("W");
    netdef.add_external_input("B");
    netdef.add_external_output("Y_cpu");
  }

  ws.RunNetOnce(netdef);
  const auto& t_cpu = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // cpu

  // NN API
  netdef.mutable_op(0)->set_output(0, "Y_nn");
  netdef.set_external_output(0, "Y_nn");
  NetDef initNet;
  NNApi model(initNet, netdef, &ws);
  std::vector<TensorCPU*> inputs, outputs;
  inputs.push_back(ws.GetBlob("X_cpu")->GetMutable<TensorCPU>());
  EXPECT_TRUE(model.run(inputs, &outputs));
  const auto& t_nn = *outputs[0];

  checkError(t_cpu, t_nn, 0.01);
}

static void test_depthwise_conv_NHWC(
    int N,
    int C,
    int H,
    int W,
    int D,
    int kernel,
    int pad_t,
    int pad_l,
    int pad_b,
    int pad_r,
    int stride_h,
    int stride_w) {
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, H, W, C);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }
  {
    auto* t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
    t->Resize(1, kernel, kernel, D);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }
  {
    auto* t = ws.CreateBlob("B")->GetMutable<TensorCPU>();
    t->Resize(D);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef_cpu;
  {
    // X: NHWC -> NCHW
    {
      auto& op = *(netdef_cpu.add_op());
      op.set_type("Transpose");
      op.add_input("X_cpu");
      op.add_output("X_t");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("axes");
        arg.add_ints(0);
        arg.add_ints(3);
        arg.add_ints(1);
        arg.add_ints(2);
      }
    }
    // X: MHWC -> CMHW
    {
      auto& op = *(netdef_cpu.add_op());
      op.set_type("Transpose");
      op.add_input("W");
      op.add_output("W_t");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("axes");
        arg.add_ints(3);
        arg.add_ints(0);
        arg.add_ints(1);
        arg.add_ints(2);
      }
    }
    {
      auto& op = *(netdef_cpu.add_op());
      op.set_type("Conv");
      op.add_input("X_t");
      op.add_input("W_t");
      op.add_input("B");
      op.add_output("Y_t");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("order");
        arg.set_s("NCHW");
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("kernel");
        arg.set_i(kernel);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_t");
        arg.set_i(pad_t);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_l");
        arg.set_i(pad_l);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_b");
        arg.set_i(pad_b);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_r");
        arg.set_i(pad_r);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("stride_h");
        arg.set_i(stride_h);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("stride_w");
        arg.set_i(stride_w);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("group");
        arg.set_i(C);
      }
    }
    // Y: NCHW -> NHWC
    {
      auto& op = *(netdef_cpu.add_op());
      op.set_type("Transpose");
      op.add_input("Y_t");
      op.add_output("Y_cpu");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("axes");
        arg.add_ints(0);
        arg.add_ints(2);
        arg.add_ints(3);
        arg.add_ints(1);
      }
    }
    netdef_cpu.add_external_input("X_cpu");
    netdef_cpu.add_external_input("W");
    netdef_cpu.add_external_input("B");
    netdef_cpu.add_external_output("Y_cpu");
  }

  NetDef netdef;
  {
    {
      auto& op = *(netdef.add_op());
      op.set_type("Conv");
      op.add_input("X_cpu");
      op.add_input("W");
      op.add_input("B");
      op.add_output("Y_nn");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("order");
        arg.set_s("NHWC");
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("kernel");
        arg.set_i(kernel);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_t");
        arg.set_i(pad_t);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_l");
        arg.set_i(pad_l);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_b");
        arg.set_i(pad_b);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_r");
        arg.set_i(pad_r);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("stride_h");
        arg.set_i(stride_h);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("stride_w");
        arg.set_i(stride_w);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("group");
        arg.set_i(C);
      }
    }
    netdef.add_external_input("X_cpu");
    netdef.add_external_input("W");
    netdef.add_external_input("B");
    netdef.add_external_output("Y_nn");
  }

  ws.RunNetOnce(netdef_cpu);
  const auto& t_cpu = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // cpu

  // NN API
  NetDef initNet;
  NNApi model(initNet, netdef, &ws);
  std::vector<TensorCPU*> inputs, outputs;
  inputs.push_back(ws.GetBlob("X_cpu")->GetMutable<TensorCPU>());
  EXPECT_TRUE(model.run(inputs, &outputs));
  const auto& t_nn = *outputs[0];

  checkError(t_cpu, t_nn, 0.01);
}

static void test_pooling(
    std::string type,
    int N,
    int C,
    int H,
    int W,
    int kernel,
    int pad_t,
    int pad_l,
    int pad_b,
    int pad_r,
    int stride_h,
    int stride_w) {
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, H, W, C);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    {
      auto& op = *(netdef.add_op());
      op.set_type(type);
      op.add_input("X_cpu");
      op.add_output("Y_cpu");
      {
        auto& arg = *(op.add_arg());
        arg.set_name("order");
        arg.set_s("NHWC");
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("kernel");
        arg.set_i(kernel);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_t");
        arg.set_i(pad_t);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_l");
        arg.set_i(pad_l);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_b");
        arg.set_i(pad_b);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("pad_r");
        arg.set_i(pad_r);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("stride_h");
        arg.set_i(stride_h);
      }
      {
        auto& arg = *(op.add_arg());
        arg.set_name("stride_w");
        arg.set_i(stride_w);
      }
    }
    netdef.add_external_input("X_cpu");
    netdef.add_external_output("Y_cpu");
  }

  ws.RunNetOnce(netdef);
  const auto& t_cpu = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // cpu

  // NN API
  netdef.mutable_op(0)->set_output(0, "Y_nn");
  netdef.set_external_output(0, "Y_nn");
  NetDef initNet;
  NNApi model(initNet, netdef, &ws);
  std::vector<TensorCPU*> inputs, outputs;
  inputs.push_back(ws.GetBlob("X_cpu")->GetMutable<TensorCPU>());
  EXPECT_TRUE(model.run(inputs, &outputs));
  const auto& t_nn = *outputs[0];

  checkError(t_cpu, t_nn, 0.01);
}

static void test_softmax(int N, int C, int H = 1, int W = 1) {
  Workspace ws;
  {
    auto* t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    if (H == 1 && W == 1) {
      t->Resize(N, C);
    } else {
      t->Resize(N, H, W, C);
    }
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(
        t->size(), 0, 30, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    {
      auto& op = *(netdef.add_op());
      op.set_type("Softmax");
      op.add_input("X_cpu");
      op.add_output("Y_cpu");
    }
    netdef.add_external_input("X_cpu");
    netdef.add_external_output("Y_cpu");
  }

  ws.RunNetOnce(netdef);
  const auto& t_cpu = ws.GetBlob("Y_cpu")->Get<TensorCPU>(); // cpu

  // NN API
  netdef.mutable_op(0)->set_output(0, "Y_nn");
  netdef.set_external_output(0, "Y_nn");
  NetDef initNet;
  NNApi model(initNet, netdef, &ws);
  std::vector<TensorCPU*> inputs, outputs;
  inputs.push_back(ws.GetBlob("X_cpu")->GetMutable<TensorCPU>());
  EXPECT_TRUE(model.run(inputs, &outputs));
  const auto& t_nn = *outputs[0];

  checkError(t_cpu, t_nn, 0.01);
}

TEST(NNApi, TestConv) {
  for (int C : {13, 32}) {
    for (int M : {4, 7, 17}) {
      for (int W : {13, 104}) {
        for (int K : {1, 3, 5}) {
          for (int P : {0, K - 1}) {
            for (int S : {1, 2}) {
              test_conv_NHWC(1, C, W, W, M, K, P, P, P, P, S, S);
            }
          }
        }
      }
    }
  }
  // Test for asymmetric padding
  // NN API only supports stride_x == stride_y
  test_conv_NHWC(1, 3, 26, 26, 7, 3, 1, 2, 2, 0, 1, 1);
  test_conv_NHWC(1, 3, 26, 26, 7, 3, 1, 2, 0, 2, 2, 2);
  test_conv_NHWC(1, 3, 26, 26, 7, 3, 1, 1, 2, 1, 2, 2);
  test_conv_NHWC(1, 3, 26, 26, 7, 3, 1, 1, 0, 1, 1, 1);
}

TEST(NNApi, Depthwise) {
  for (int C : {13, 32}) {
    for (int W : {13, 104}) {
      for (int K : {1, 3}) {
        for (int P : {0, K - 1}) {
          for (int S : {1, 2}) {
            test_depthwise_conv_NHWC(1, C, W, W, C, K, P, P, P, P, S, S);
          }
        }
      }
    }
  }
  // Test for asymmetric padding
  // NN API only supports stride_x == stride_y
  test_depthwise_conv_NHWC(1, 3, 26, 26, 3, 3, 1, 2, 2, 0, 1, 1);
  test_depthwise_conv_NHWC(1, 3, 26, 26, 3, 3, 1, 2, 0, 2, 2, 2);
  test_depthwise_conv_NHWC(1, 3, 26, 26, 3, 3, 1, 1, 2, 1, 2, 2);
  test_depthwise_conv_NHWC(1, 3, 26, 26, 3, 3, 1, 1, 0, 1, 1, 1);
}

TEST(NNApi, TestRelu) {
  test_relu(1, 4, 10, 10);
  test_relu(1, 16, 128, 128);
}

TEST(NNApi, TestAveragePool) {
  for (int C : {13, 32}) {
    for (int W : {13, 104}) {
      for (int K : {1, 3}) {
        for (int P : {0, K - 1}) {
          for (int S : {1, 2}) {
            test_pooling("AveragePool", 1, C, W, W, K, P, P, P, P, S, S);
          }
        }
      }
    }
  }
  test_pooling("AveragePool", 1, 3, 26, 26, 3, 1, 2, 2, 0, 1, 1);
  test_pooling("AveragePool", 1, 3, 26, 26, 3, 1, 2, 0, 2, 2, 2);
  test_pooling("AveragePool", 1, 3, 26, 26, 3, 1, 1, 2, 1, 2, 2);
  test_pooling("AveragePool", 1, 3, 26, 26, 3, 1, 1, 0, 1, 1, 1);
}

TEST(NNApi, TestMaxPool) {
  for (int C : {13, 32}) {
    for (int W : {13, 104}) {
      for (int K : {1, 3}) {
        for (int P : {0, K - 1}) {
          for (int S : {1, 2}) {
            test_pooling("MaxPool", 1, C, W, W, K, P, P, P, P, S, S);
          }
        }
      }
    }
  }
  test_pooling("MaxPool", 1, 3, 26, 26, 3, 1, 2, 2, 0, 1, 1);
  test_pooling("MaxPool", 1, 3, 26, 26, 3, 1, 2, 0, 2, 2, 2);
  test_pooling("MaxPool", 1, 3, 26, 26, 3, 1, 1, 2, 1, 2, 2);
  test_pooling("MaxPool", 1, 3, 26, 26, 3, 1, 1, 0, 1, 1, 1);
}

TEST(NNApi, TestSoftmax) {
  test_softmax(1, 100);
  test_softmax(2, 17);

  // NN API doesn't seem to work for 4D tensor
  // test_softmax(1, 100, 13, 13);
  // test_softmax(5, 17, 13, 13);
}

} // namespace

} // namespace caffe2
