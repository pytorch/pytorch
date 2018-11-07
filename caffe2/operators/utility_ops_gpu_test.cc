#include <iostream>

#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/flags.h"
#include "caffe2/operators/utility_ops.h"
#include <gtest/gtest.h>

C10_DECLARE_string(caffe_test_root);

namespace caffe2 {

static void AddConstInput(
    const vector<int64_t>& shape,
    const float value,
    const string& name,
    Workspace* ws) {
  DeviceOption option;
  option.set_device_type(PROTO_CUDA);
  CUDAContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = BlobGetMutableTensor(blob, CUDA);
  tensor->Resize(shape);
  math::Set<float, CUDAContext>(
      tensor->numel(), value, tensor->template mutable_data<float>(), &context);
  return;
}

TEST(UtilityOpGPUTest, testReshapeWithScalar) {
  if (!HasCudaGPU())
    return;
  Workspace ws;
  OperatorDef def;
  def.set_name("test_reshape");
  def.set_type("Reshape");
  def.add_input("X");
  def.add_output("XNew");
  def.add_output("OldShape");
  def.add_arg()->CopyFrom(MakeArgument("shape", vector<int64_t>{1}));
  def.mutable_device_option()->set_device_type(PROTO_CUDA);
  AddConstInput(vector<int64_t>(), 3.14, "X", &ws);
  // execute the op
  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_TRUE(op->Run());
  Blob* XNew = ws.GetBlob("XNew");
  const Tensor& XNewTensor = XNew->Get<Tensor>();
  EXPECT_EQ(1, XNewTensor.dim());
  EXPECT_EQ(1, XNewTensor.numel());
}

} // namespace caffe2
