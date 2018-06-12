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

#define EXPECT_TENSOR_EQ(_YA, _YE)                                     \
  do {                                                                 \
    EXPECT_TRUE((_YA).dims() == (_YE).dims());                         \
    for (auto i = 0; i < (_YA).size(); ++i) {                          \
      EXPECT_FLOAT_EQ((_YA).data<float>()[i], (_YE).data<float>()[i]); \
    }                                                                  \
  } while (0);

TEST(UtilityOpTest, testIdentity) {
  Workspace ws;
  OperatorDef def;
  def.set_name("test_identity");
  def.set_type("Identity");
  def.add_input("X");
  def.add_output("Y");
  AddConstInput(vector<TIndex>{1, 2, 3}, 1.23, "X", &ws);
  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_TRUE(op->Run());
  Blob* X = ws.GetBlob("X");
  Blob* Y = ws.GetBlob("Y");
  const TensorCPU& XTensor = X->Get<TensorCPU>();
  const TensorCPU& YTensor = Y->Get<TensorCPU>();
  EXPECT_TENSOR_EQ(XTensor, YTensor);
}
#undef EXPECT_TENSOR_EQ

} // namespace caffe2
