#include <iostream>

#include "caffe2/core/flags.h"
#include "caffe2/operators/utility_ops.h"
#include <gtest/gtest.h>

CAFFE2_DECLARE_string(caffe_test_root);

namespace caffe2 {

static void AddConstInput(
    const vector<TIndex>& shape,
    const float value,
    const string& name,
    Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(shape);
  math::Set<float, CPUContext>(
      tensor->size(), value, tensor->mutable_data<float>(), &context);
  return;
}

TEST(UtilityOpTest, testEnsureCPUOutput) {
  Workspace ws;
  OperatorDef def;
  def.set_name("test");
  def.set_type("EnsureCPUOutput");
  def.add_input("X");
  def.add_output("Y");
  AddConstInput(vector<TIndex>{5, 10}, 3.14, "X", &ws);
  Blob* Xblob = ws.GetBlob("X");
  EXPECT_NE(nullptr, Xblob);
  // input X should be a CPUTensor
  EXPECT_TRUE(Xblob->IsType<Tensor<CPUContext>>());
  // now execute the op to get Y
  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());
  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  // output Y should be a CPUTensor
  EXPECT_TRUE(Yblob->IsType<Tensor<CPUContext>>());
  const TensorCPU& Y_cpu = Yblob->Get<Tensor<CPUContext>>();
  EXPECT_EQ(Y_cpu.size(), 5 * 10);
  for (int i = 0; i < Y_cpu.size(); ++i) {
    EXPECT_LT(Y_cpu.data<float>()[i], 3.15);
    EXPECT_GT(Y_cpu.data<float>()[i], 3.13);
  }
}

TEST(UtilityOpTest, testReshapeWithScalar) {
  Workspace ws;
  OperatorDef def;
  def.set_name("test_reshape");
  def.set_type("Reshape");
  def.add_input("X");
  def.add_output("XNew");
  def.add_output("OldShape");
  def.add_arg()->CopyFrom(MakeArgument("shape", vector<int64_t>{1}));
  AddConstInput(vector<TIndex>(), 3.14, "X", &ws);
  // execute the op
  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_TRUE(op->Run());
  Blob* XNew = ws.GetBlob("XNew");
  const TensorCPU& XNewTensor = XNew->Get<Tensor<CPUContext>>();
  EXPECT_EQ(1, XNewTensor.ndim());
  EXPECT_EQ(1, XNewTensor.size());
}

} // namespace caffe2
