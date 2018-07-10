#include <string>

#include <gtest/gtest.h>
#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

class JustTest : public OperatorBase
{
    public:
    using OperatorBase::OperatorBase;
    bool Run(int /* unused */ /*stream_id*/) override { return true; }
    virtual std::string type() { return "BASE"; }
};

class JustTestHIP : public JustTest
{
    public:
    using JustTest::JustTest;
    bool Run(int /* unused */ /*stream_id*/) override { return true; }
    std::string type() override { return "HIP"; }
};

class JustTestMIOPEN : public JustTest {
 public:
  using JustTest::JustTest;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  std::string type() override {
    return "MIOPEN";
  }
};

OPERATOR_SCHEMA(JustTest).NumInputs(0, 1).NumOutputs(0, 1);
REGISTER_HIP_OPERATOR(JustTest, JustTestHIP);
REGISTER_MIOPEN_OPERATOR(JustTest, JustTestMIOPEN);

TEST(EnginePrefTest, GPUDeviceDefaultPreferredEngines)
{
    if(!HasHipGPU())
        return;
    OperatorDef op_def;
    Workspace ws;
    op_def.mutable_device_option()->set_device_type(HIP);
    op_def.set_type("JustTest");

    {
      const auto op = CreateOperator(op_def, &ws);
      EXPECT_NE(nullptr, op.get());
      EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "HIP");
    }
}

} // namespace caffe2
