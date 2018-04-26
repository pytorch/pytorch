#include <string>

#include <gtest/gtest.h>
#include "caffe2/core/common_hip.h"
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

#if 0 // Ashish TBD: Add MIOpen here
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
#endif

OPERATOR_SCHEMA(JustTest).NumInputs(0, 1).NumOutputs(0, 1);
REGISTER_HIP_OPERATOR(JustTest, JustTestHIP);
#if 0 // Ashish TBD: MIOpen goes here
REGISTER_CUDNN_OPERATOR(JustTest, JustTestCUDNN);
#endif

TEST(EnginePrefTest, GPUDeviceDefaultPreferredEngines)
{
    if(!HasHipGPU())
        return;
    OperatorDef op_def;
    Workspace ws;
    op_def.mutable_device_option()->set_device_type(HIP);
    op_def.set_type("JustTest");

#if 0 // Ashish TBD: MIOpen here
  {
    const auto op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
    // CUDNN should be taken as it's in the default global preferred engines
    // list
    EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "CUDNN");
  }
#endif
}

} // namespace caffe2
