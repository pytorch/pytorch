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

OPERATOR_SCHEMA(JustTest).NumInputs(0, 1).NumOutputs(0, 1);
REGISTER_HIP_OPERATOR(JustTest, JustTestHIP);

TEST(EnginePrefTest, GPUDeviceDefaultPreferredEngines)
{
    if(!HasHipGPU())
        return;
    OperatorDef op_def;
    Workspace ws;
    op_def.mutable_device_option()->set_device_type(HIP);
    op_def.set_type("JustTest");
}

} // namespace caffe2
