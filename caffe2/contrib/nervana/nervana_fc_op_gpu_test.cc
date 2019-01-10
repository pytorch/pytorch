#include "nervana.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/workspace.h"
#include "caffe2/operators/fully_connected_op.h"
#include "caffe2/utils/math.h"
#include "common/gtest/gtest_extensions.h"

#include <gtest/gtest.h>

CAFFE2_DECLARE_string(caffe_test_root);

namespace caffe2 {

namespace {
static void AddConstInput(const std::vector<int>& shape, const float value,
                          const string& name, Workspace* ws) {
  DeviceOption option;
  option.set_device_type(CUDA);
  CUDAContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<Tensor<CUDAContext>>();
  tensor->Resize(shape);
  math::Set<float, CUDAContext>(tensor->size(), value,
                                tensor->mutable_data<float>(),
                                &context);
  return;
}
}  // namespace

TEST(NervanaFullyConnectedTest, Test) {
  if (!NervanaKernelLoaded()) {
    SKIP() << "Nervana kernels are not loaded. Skipping test.";
  }
  Workspace ws;
  OperatorDef def;
  def.set_name("test");
  def.set_type("FC");
  def.add_input("X");
  def.add_input("W");
  def.add_input("B");
  def.add_output("Y");
  def.mutable_device_option()->set_device_type(CUDA);
  def.set_engine("NERVANA");
  AddConstInput(std::vector<int>{5, 10}, 1., "X", &ws);
  AddConstInput(std::vector<int>{6, 10}, 1., "W", &ws);
  AddConstInput(std::vector<int>{6}, 0.1, "B", &ws);
  unique_ptr<OperatorBase> op(
      new FullyConnectedOp<CUDAContext, NervanaEngine>(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());
  Blob* Yblob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Yblob);
  auto& Y = Yblob->Get<Tensor<CUDAContext>>();
  TensorCPU Y_cpu(Y);
  EXPECT_EQ(Y.size(), 5 * 6);
  for (int i = 0; i < Y.size(); ++i) {
    CHECK_LT(Y_cpu.data<float>()[i], 10.11);
    CHECK_GT(Y_cpu.data<float>()[i], 10.09);
  }
}

}  // namespace caffe2
