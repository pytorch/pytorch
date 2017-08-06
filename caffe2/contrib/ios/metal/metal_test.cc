// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/core/logging.h"
#include "caffe2/core/workspace.h"
#include "metal_test.h"
#include "rewrite_net.h"

#define DEBUGGING false

namespace caffe2 {
void testMetalCopyOps(int N, int C, int H, int W, float error) {
  LOG(INFO) << "MetalCopyFrom/To Test";
  Workspace ws;
  {
    auto *t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyToMetalGPU");
    op.add_input("X_cpu");
    op.add_output("X_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyFromMetalGPU");
    op.add_input("X_mtl");
    op.add_output("Y_cpu");
  }

  ws.RunNetOnce(netdef);
  const auto &t1 = ws.GetBlob("X_cpu")->Get<TensorCPU>();
  const auto &t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
  CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());

//  for (auto i = 0; i < t1.size(); ++i) {
//    LOG(INFO) << "i: " << i << ", CPU: " << t1.data<float>()[i] << ", MTL: " << t2.data<float>()[i];
//  }

  for (auto i = 0; i < t1.size(); ++i) {
    CHECK_NEAR(t1.data<float>()[i], t2.data<float>()[i], error);
  }
}

void testMetalInstanceNorm(int N, int C, int H, int W, float error) {
  LOG(INFO) << "MetalInstanceNorm Test";
  Workspace ws;
  {
    auto *t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(t->size(), 0, 3, t->mutable_data<float>(), &ctx);
  }

  {
    auto *t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
    t->Resize(C);
    CPUContext ctx;
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = i;
    }
  }
  {
    auto *t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
    t->Resize(C);
    CPUContext ctx;
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = 8 - 2 * i;
    }
  }

  NetDef netdef;
  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyToMetalGPU");
    op.add_input("X_cpu");
    op.add_output("X_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyToMetalGPUFloat16");
    op.add_input("W");
    op.add_output("W_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyToMetalGPUFloat16");
    op.add_input("b");
    op.add_output("b_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("MetalInstanceNorm");
    op.add_input("X_mtl");
    op.add_input("W_mtl");
    op.add_input("b_mtl");
    op.add_output("Y_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyFromMetalGPU");
    op.add_input("Y_mtl");
    op.add_output("Y_cpu");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("InstanceNorm");
    op.add_input("X_cpu");
    op.add_input("W");
    op.add_input("b");
    auto &arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto &t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
  const auto &t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

  CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
  for (auto i = 0; i < t1.size(); ++i) {
    const float t1_i = t1.data<float>()[i];
    const float t2_i = t2.data<float>()[i];
    CHECK_NEAR(t1_i, t2_i, error);
  }
}

void testMetalPRelu(int N, int C, int H, int W, int K, float error) {
  LOG(INFO) << "MetalPRelu Test";
  Workspace ws;
  {
    auto *t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  {
    auto *t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
    t->Resize(K);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  // The MetalPRelu is an in-place operator
  NetDef netdef;
  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyToMetalGPU");
    op.add_input("X_cpu");
    op.add_output("X_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyToMetalGPUFloat16");
    op.add_input("b");
    op.add_output("b_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("MetalPRelu");
    op.add_input("X_mtl");
    op.add_input("b_mtl");
    op.add_output("X_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyFromMetalGPU");
    op.add_input("X_mtl");
    op.add_output("Y_cpu");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("PRelu");
    op.add_input("X_cpu");
    op.add_input("b");
    auto &arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto &t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
  const auto &t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

  CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());

  for (auto i = 0; i < t1.size(); ++i) {
    const float t1_i = t1.data<float>()[i];
    const float t2_i = t2.data<float>()[i];
    CHECK_NEAR(t1_i, t2_i, error);
  }
}

void testMetalInstanceNormPRelu(int N, int C, int H, int W, float error) {
  LOG(INFO) << "MetalInstanceNormPRelu Test";
  Workspace ws;
  {
    auto *t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    // Too noisy.
    math::RandGaussian<float, CPUContext>(t->size(), 0, 3, t->mutable_data<float>(), &ctx);
  }

  {
    auto *t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
    t->Resize(C);
    CPUContext ctx;
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = i;
    }
  }
  {
    auto *t = ws.CreateBlob("b1")->GetMutable<TensorCPU>();
    t->Resize(C);
    CPUContext ctx;
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = 8 - 2 * i;
    }
  }
  // bias for PRelu
  {
    auto *t = ws.CreateBlob("b2")->GetMutable<TensorCPU>();
    t->Resize(C);
    CPUContext ctx;

    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyToMetalGPU");
    op.add_input("X_cpu");
    op.add_output("X_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyToMetalGPUFloat16");
    op.add_input("W");
    op.add_output("W_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyToMetalGPUFloat16");
    op.add_input("b1");
    op.add_output("b1_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyToMetalGPUFloat16");
    op.add_input("b2");
    op.add_output("b2_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("MetalInstanceNormPRelu");
    op.add_input("X_mtl");
    op.add_input("W_mtl");
    op.add_input("b1_mtl");
    op.add_input("b2_mtl");
    op.add_output("Y_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyFromMetalGPU");
    op.add_input("Y_mtl");
    op.add_output("Y_cpu");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("InstanceNorm");
    op.add_input("X_cpu");
    op.add_input("W");
    op.add_input("b1");
    auto &arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_mid");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("PRelu");
    op.add_input("Y_mid");
    op.add_input("b2");
    auto &arg = *(op.add_arg());
    arg.set_name("order");
    arg.set_s("NCHW");
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto &t1 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
  const auto &t2 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

  CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());

  for (auto i = 0; i < t1.size(); ++i) {
    const float t1_i = t1.data<float>()[i];
    const float t2_i = t2.data<float>()[i];
    CHECK_NEAR(t1_i, t2_i, error);
  }
}

void testMetalConv(int N, int C, int H, int W, int K, int kernel_h, int kernel_w, int pad, float error) {
  LOG(INFO) << "MetalConv Test";
  Workspace ws;
  {
    auto *t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  {
    auto *t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
    t->Resize(K, C, kernel_h, kernel_w);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  {
    auto *t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
    t->Resize(K);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyToMetalGPU");
    op.add_input("X_cpu");
    op.add_output("X_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyWeightTensorToMetalGPU");
    op.add_input("W");
    op.add_output("W_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("MetalConv");
    op.add_input("X_mtl");
    op.add_input("W_mtl");
    op.add_input("b");
    op.set_engine("METAL");
    {
      auto &arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
    {
      auto &arg = *(op.add_arg());
      arg.set_name("kernel");
      arg.set_i(kernel_h);
    }
    {
      auto &arg = *(op.add_arg());
      arg.set_name("pad");
      arg.set_i(pad);
    }
    op.add_output("Y_mtl");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyFromMetalGPU");
    op.add_input("Y_mtl");
    op.add_output("Y_cpu");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("Conv");
    op.add_input("X_cpu");
    op.add_input("W");
    op.add_input("b");
    {
      auto &arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
    {
      auto &arg = *(op.add_arg());
      arg.set_name("kernel");
      arg.set_i(kernel_h);
    }
    {
      auto &arg = *(op.add_arg());
      arg.set_name("pad");
      arg.set_i(pad);
    }
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto &t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
  const auto &t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();

  CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());

//  for (auto i = 0; i < t1.size(); ++i) {
//    const float t1_i = t1.data<float>()[i];
//    const float t2_i = t2.data<float>()[i];
//    if (std::abs(t1_i - t2_i) > 0.5) {
//      LOG(INFO) << "i: " << i << ", CPU: " << t1_i << ", MTL: " << t2_i << ", error: " << std::abs(t1_i - t2_i) / t1_i;
//    }
//  }

  for (auto i = 0; i < t1.size(); ++i) {
    // FP16 <-> FP32 round trip, accumulation, etc.
    const float t1_i = t1.data<float>()[i];
    const float t2_i = t2.data<float>()[i];
    CHECK_NEAR(t1_i, t2_i, error);
  }
}

void testMetalConvTranspose(
    int N, int C, int H, int W, int K, int kernel_h, int kernel_w, int pad, int stride, float error) {
  LOG(INFO) << "MetalConvTranspose Test";
  Workspace ws;
  {
    auto *t = ws.CreateBlob("X_cpu")->GetMutable<TensorCPU>();
    t->Resize(N, C, H, W);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  {
    auto *t = ws.CreateBlob("W")->GetMutable<TensorCPU>();
    t->Resize(K, C, kernel_h, kernel_w);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  {
    auto *t = ws.CreateBlob("b")->GetMutable<TensorCPU>();
    t->Resize(C);
    CPUContext ctx;
    math::RandGaussian<float, CPUContext>(t->size(), 0, 1, t->mutable_data<float>(), &ctx);
  }

  NetDef netdef;
  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyToMetalGPU");
    op.add_input("X_cpu");
    op.add_output("X_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyTransposeWeightTensorToMetalGPU");
    op.add_input("W");
    op.add_output("W_mtl");
    op.set_engine("METAL");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("MetalConvTranspose");
    op.add_input("X_mtl");
    op.add_input("W_mtl");
    op.add_input("b");
    op.set_engine("METAL");
    {
      auto &arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
    {
      auto &arg = *(op.add_arg());
      arg.set_name("kernel");
      arg.set_i(kernel_h);
    }
    {
      auto &arg = *(op.add_arg());
      arg.set_name("pad");
      arg.set_i(pad);
    }
    {
      auto &arg = *(op.add_arg());
      arg.set_name("stride");
      arg.set_i(stride);
    }
    op.add_output("Y_mtl");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("CopyFromMetalGPU");
    op.add_input("Y_mtl");
    op.add_output("Y_cpu");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("ConvTranspose");
    op.add_input("X_cpu");
    op.add_input("W");
    op.add_input("b");
    {
      auto &arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
    {
      auto &arg = *(op.add_arg());
      arg.set_name("kernel");
      arg.set_i(kernel_h);
    }
    {
      auto &arg = *(op.add_arg());
      arg.set_name("pad");
      arg.set_i(pad);
    }
    {
      auto &arg = *(op.add_arg());
      arg.set_name("stride");
      arg.set_i(stride);
    }
    op.add_output("Y_ref");
  }

  ws.RunNetOnce(netdef);
  const auto &t2 = ws.GetBlob("Y_cpu")->Get<TensorCPU>();
  const auto &t1 = ws.GetBlob("Y_ref")->Get<TensorCPU>();
  CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
  LOG(INFO) << "N: " << t1.dim(0) << " C: " << t1.dim(1) << " H: " << t1.dim(2) << " W: " << t1.dim(3);

#if DEBUGGING
  for (auto i = 0; i < t1.size(); ++i) {
    const float t1_i = t1.data<float>()[i];
    const float t2_i = t2.data<float>()[i];
    if (std::abs(t1_i - t2_i) > error) {
      LOG(INFO) << "i: " << i << ", CPU: " << t1_i << ", MTL: " << t2_i
                << ", relative error: " << 100 * std::abs(t1_i - t2_i) / t1_i << "%";
    }
  }

  printf("CPU:");
  for (int i = 0; i < t1.size(); i++) {
    const float t1_i = t1.data<float>()[i];
    if (i % t1.dim(2) == 0)
      printf("\n");
    printf("%f\t", t1_i);
  }

  printf("\nMETAL:");
  for (int i = 0; i < t1.size(); i++) {
    const float t2_i = t2.data<float>()[i];
    if (i % t2.dim(2) == 0)
      printf("\n");
    printf("%f\t", t2_i);
  }
  printf("\n");
#endif

  for (auto i = 0; i < t1.size(); ++i) {
    // FP16 <-> FP32 round trip, accumulation, etc.
    const float t1_i = t1.data<float>()[i];
    const float t2_i = t2.data<float>()[i];
    CHECK_NEAR(t1_i, t2_i, error);
  }
}
void testMetalRewriteWithFusion() {
  for (const auto &computeOp : std::vector<std::string>{"InstanceNorm"}) {
    LOG(INFO) << "RewriteForMetal Fusion/Copy Test";
    NetDef netdef;
    netdef.add_external_input("X");
    netdef.add_external_output("Y");
    // These two ops can be fused.
    {
      auto &op = *(netdef.add_op());
      op.set_type(computeOp);
      op.add_input("X");
      op.add_input("W");
      op.add_input("b1");
      op.add_output("Y");
    }
    {
      auto &op = *(netdef.add_op());
      op.set_type("PRelu");
      op.add_input("Y");
      op.add_input("b2");
      op.add_output("Y");
    }
    // Can't fuse these as not in-place (can fix by using SSA).
    {
      auto &op = *(netdef.add_op());
      op.set_type(computeOp);
      op.add_input("X2");
      op.add_input("W");
      op.add_input("b1");
      op.add_output("Y2");
    }
    {
      auto &op = *(netdef.add_op());
      op.set_type("PRelu");
      op.add_input("Y2");
      op.add_input("b2");
      op.add_output("Y");
    }

    netdef = rewritePredictNetForMetal(netdef, "METAL");
    // dumpDef(netdef);

    auto ty = [&](size_t i) { return netdef.op(i).type(); };
    auto i0 = [&](size_t i) { return netdef.op(i).input(0); };
    auto o0 = [&](size_t i) { return netdef.op(i).output(0); };
    CHECK_EQ(netdef.op_size(), 5);
    CHECK_EQ(ty(0), "CopyToMetalGPU");
    CHECK_EQ(ty(1), std::string("Metal") + computeOp + std::string("PRelu"));
    CHECK_EQ(ty(2), std::string("Metal") + computeOp);
    CHECK_EQ(ty(3), "MetalPRelu");
    CHECK_EQ(ty(4), "CopyFromMetalGPU");
    CHECK_EQ(i0(0), "X");
    CHECK_EQ(i0(1), o0(0));
    CHECK_EQ(o0(2), "Y2_M");
    CHECK_EQ(i0(3), o0(2));
    CHECK_EQ(i0(4), o0(3));
    CHECK_NE(o0(4), i0(4));
    CHECK_EQ(netdef.external_input(0), "X");
    CHECK_EQ(netdef.external_output(0), "Y");
  }
}

void testMetalRewriteWithMultiInputCPUOps() {

  LOG(INFO) << "RewriteForMetal Test";
  NetDef netdef;
  netdef.add_external_input("X");
  netdef.add_external_output("Y");
  // These two ops can be fused.
  {
    auto &op = *(netdef.add_op());
    op.set_type("Conv");
    op.add_input("X");
    op.add_input("W");
    op.add_input("b");
    op.add_output("Y1");
  }
  {
    auto &op = *(netdef.add_op());
    op.set_type("Conv");
    op.add_input("Y1");
    op.add_input("W");
    op.add_input("b");
    op.add_output("Y2");
  }
  {
    auto &op = *(netdef.add_op());
    op.set_type("Add");
    op.add_input("Y1");
    op.add_input("Y2");
    op.add_output("Y");
  }

  netdef = rewritePredictNetForMetal(netdef, "METAL");
  // dumpDef(netdef);

  auto ty = [&](size_t i) { return netdef.op(i).type(); };
  auto i0 = [&](size_t i) { return netdef.op(i).input(0); };
  auto i1 = [&](size_t i) { return netdef.op(i).input(1); };
  auto o0 = [&](size_t i) { return netdef.op(i).output(0); };
  CHECK_EQ(netdef.op_size(), 6);
  CHECK_EQ(ty(0), "CopyToMetalGPU");
  CHECK_EQ(ty(1), "MetalConv");
  CHECK_EQ(ty(2), "MetalConv");
  CHECK_EQ(ty(3), "CopyFromMetalGPU");
  CHECK_EQ(ty(4), "CopyFromMetalGPU");
  CHECK_EQ(ty(5), "Add");
  CHECK_EQ(i0(0), "X");
  CHECK_EQ(i0(1), o0(0));
  CHECK_EQ(o0(2), "Y2_M");
  CHECK_EQ(i0(3), o0(1));
  CHECK_EQ(i0(4), o0(2));
  CHECK_EQ(i0(5), o0(3));
  CHECK_EQ(i1(5), o0(4));
  CHECK_EQ(o0(5), "Y");
  CHECK_EQ(netdef.external_input(0), "X");
  CHECK_EQ(netdef.external_output(0), "Y");
}

void testMetalRewriteFailure() {
  LOG(INFO) << "RewriteForMetal Failure Test";
  NetDef netdef;
  netdef.add_external_input("X");
  netdef.add_external_output("Y");
  {
    auto &op = *(netdef.add_op());
    op.set_type("Conv");
    op.add_input("X");
    op.add_input("W");
    op.add_input("b");
    op.add_output("Y1");
  }
  {
    auto &op = *(netdef.add_op());
    op.set_type("Conv");
    op.add_input("X");
    op.add_input("W");
    op.add_input("b");
    op.add_output("Y2");
  }

  {
    auto &op = *(netdef.add_op());
    op.set_type("Concat");
    op.add_input("Y1");
    op.add_input("Y2");
    op.add_output("Y");
  }
  try {;
    netdef = rewritePredictNetForMetal(netdef, "METAL");
    // dumpDef(netdef);
    CHECK(false) << "Shouldn't reach here, due to multiple usages of X";
  } catch (const std::exception &e) {
    LOG(INFO) << "RewriteForMetal failed";
  }
}

void testMetal() {
  testMetalCopyOps(1, 3, 2, 1, 1e-2);
  testMetalCopyOps(1, 4, 1, 1, 1e-2);
  testMetalCopyOps(1, 4, 8, 3, 1e-2);
  testMetalCopyOps(1, 6, 8, 3, 1e-2);
  testMetalCopyOps(1, 4, 1, 2, 1e-2);
  testMetalCopyOps(1, 8, 6, 1, 1e-2);
  testMetalCopyOps(1, 12, 13, 18, 1e-2);

  testMetalInstanceNorm(1, 3, 120, 140, 0.05);
  testMetalInstanceNorm(1, 12, 120, 140, 0.05);

  testMetalPRelu(1, 3, 8, 13, 3, 0.1);
  testMetalPRelu(1, 3, 8, 13, 1, 0.1);

  testMetalInstanceNormPRelu(1, 12, 120, 140, 0.2);

  testMetalConv(1, 12, 57, 72, 8, 3, 3, 1, 1.5);
  testMetalConv(1, 12, 57, 72, 8, 3, 3, 2, 1.5);
  testMetalConv(1, 12, 57, 72, 8, 3, 3, 0, 1.5);
  testMetalConv(1, 12, 57, 72, 8, 2, 2, 0, 1.5);

#if DEBUGGING
  testMetalConvTranspose(1, 1, 6, 6, 1, 1, 1, 0, 2, 0.1);
  testMetalConvTranspose(1, 1, 6, 6, 1, 2, 2, 0, 2, 0.1);
  testMetalConvTranspose(1, 1, 6, 6, 1, 3, 3, 0, 2, 0.1);
  testMetalConvTranspose(1, 1, 6, 6, 1, 4, 4, 0, 2, 0.1);
  testMetalConvTranspose(1, 1, 6, 6, 1, 5, 5, 0, 2, 0.1);
  testMetalConvTranspose(1, 1, 6, 6, 1, 6, 6, 0, 2, 0.1);
#endif
  
  testMetalConvTranspose(1, 16, 320, 180, 16, 2, 2, 0, 2, 1.5);
  testMetalConvTranspose(1, 4, 320, 180, 4, 4, 4, 1, 2, 1.5);
  testMetalConvTranspose(1, 4, 320, 180, 4, 4, 4, 0, 4, 1.5);

  testMetalRewriteWithFusion();
  testMetalRewriteWithMultiInputCPUOps();
  testMetalRewriteFailure();
  
}

NetDef truncateAfter(NetDef def, size_t idx) {
  // idx = 0, net = 10 -> remove 9
  // idx = 0, net = 1 -> remove 0
  const auto toRemove = def.op_size() - idx - 1;
  for (auto i = 0; i < toRemove; ++i) {
    def.mutable_op()->RemoveLast();
  }
  CHECK_EQ(def.op_size(), idx + 1);
  return def;
}

void compareModels(const NetDef &initNet, NetDef predictNet) {
  auto *arg = predictNet.mutable_op(0)->mutable_arg(0);
  CHECK_EQ(arg->name(), "noise_std");
  arg->set_f(0.000001);

  for (auto i = 0; i < predictNet.op_size(); ++i) {
    auto truncatedPredictNet = truncateAfter(predictNet, i);

    // The copyFromMetalGPUop is added in the rewriting process
    NetDef truncatedMetalPredictNet = rewritePredictNetForMetal(truncatedMetalPredictNet, "METAL");
    NetDef metalInitNet = rewriteInitNetForMetal(metalInitNet, truncatedMetalPredictNet, "METAL");

    // dumpDef(truncatedPredictNet);
    // dumpDef(truncatedMetalPredictNet);

    Workspace cws;
    cws.RunNetOnce(initNet);
    {
      auto *t = cws.CreateBlob(predictNet.external_input(0))->GetMutable<TensorCPU>();
      t->Resize(1, 224, 224, 4);
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<uint8_t>()[i] = i % 225;
      }
    }
    cws.RunNetOnce(truncatedPredictNet);

    Workspace mws;
    mws.RunNetOnce(metalInitNet);
    {
      auto *t = mws.CreateBlob(predictNet.external_input(0))->GetMutable<TensorCPU>();
      t->Resize(1, 224, 224, 4);
      for (auto i = 0; i < t->size(); ++i) {
        t->mutable_data<uint8_t>()[i] = i % 225;
      }
    }
    mws.RunNetOnce(truncatedMetalPredictNet);

    const auto name = truncatedPredictNet.op(truncatedPredictNet.op_size() - 1).output(0);

    LOG(INFO) << "Checking correspondence for name: " << name << ", idx: " << i;
    {
      const auto &mt = mws.GetBlob(name)->Get<TensorCPU>();
      const auto &ct = cws.GetBlob(name)->Get<TensorCPU>();
      CHECK_EQ(mt.dims(), ct.dims());
      for (auto j = 0; j < mt.size(); ++j) {
        if (mt.IsType<float>()) {
          if (j < 10) {
            LOG(INFO) << "i: " << i << ", j: " << j << ", CPU: " << ct.data<float>()[j]
                      << ", MTL: " << mt.data<float>()[j];
          }
          CHECK_NEAR(mt.data<float>()[j], ct.data<float>()[j], 5);
        } else {
          CHECK(mt.IsType<uint8_t>());
          if (j < 10) {
            LOG(INFO) << "i: " << i << ", j: " << j << ", CPU: " << ct.data<uint8_t>()[j]
                      << ", MTL: " << mt.data<uint8_t>()[j];
          }
          CHECK_NEAR(mt.data<uint8_t>()[j], ct.data<uint8_t>()[j], 5);
        }
      }
    }
  }
}
} // namespace caffe2
