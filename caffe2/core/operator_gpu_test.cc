#include <string>

#include <gtest/gtest.h>
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

class JustTest : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  virtual std::string type() {
    return "BASE";
  }
};

class JustTestCUDA : public JustTest {
 public:
  using JustTest::JustTest;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  std::string type() override {
    return "CUDA";
  }
};

class JustTestCUDNN : public JustTest {
 public:
  using JustTest::JustTest;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  std::string type() override {
    return "CUDNN";
  }
};

OPERATOR_SCHEMA(JustTest).NumInputs(0, 1).NumOutputs(0, 1);
REGISTER_CUDA_OPERATOR(JustTest, JustTestCUDA);
REGISTER_CUDNN_OPERATOR(JustTest, JustTestCUDNN);

TEST(EnginePrefTest, GPUDeviceDefaultPreferredEngines) {
  if (!HasCudaGPU())
    return;
  OperatorDef op_def;
  Workspace ws;
  op_def.mutable_device_option()->set_device_type(PROTO_CUDA);
  op_def.set_type("JustTest");

  {
    const auto op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
    // CUDNN should be taken as it's in the default global preferred engines
    // list
    EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "CUDNN");
  }
}

} // namespace caffe2
