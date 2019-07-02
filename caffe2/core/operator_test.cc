#include <iostream>

#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include <gtest/gtest.h>

namespace caffe2 {

// Since we instantiate this on CPU and GPU (but don't want a
// CUDAContext dependency, we use OperatorBase. In general, you only
// want to inherit from Operator<Context> in your code.
class JustTest : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  virtual string type() {
    return "base";
  }
};

class JustTestAndNeverConstructs : public JustTest {
 public:
  JustTestAndNeverConstructs(const OperatorDef& def, Workspace* ws)
      : JustTest(def, ws) {
    throw UnsupportedOperatorFeature("I just don't construct.");
  }
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  string type() override {
    return "FOO";
  }
};

class JustTestAndDoesConstruct : public JustTest {
 public:
  using JustTest::JustTest;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  string type() override {
    return "BAR";
  }
};

class JustTestWithSomeOutput : public JustTest {
 public:
  using JustTest::JustTest;
  bool Run(int /* unused */ /*stream_id*/) override {
    *OperatorBase::Output<int>(0) = 5;
    return true;
  }
  string type() override {
    return "SETTING_SOME_OUTPUT";
  }
};

OPERATOR_SCHEMA(JustTest).NumInputs(0, 1).NumOutputs(0, 1);
OPERATOR_SCHEMA(JustTestCPUOnly).NumInputs(0, 1).NumOutputs(0, 1);
OPERATOR_SCHEMA(JustTestWithSomeOutput);

REGISTER_CPU_OPERATOR(JustTest, JustTest);
REGISTER_CPU_OPERATOR(JustTestCPUOnly, JustTest);
REGISTER_CPU_OPERATOR_WITH_ENGINE(JustTest, FOO, JustTestAndNeverConstructs);
REGISTER_CPU_OPERATOR_WITH_ENGINE(JustTest, BAR, JustTestAndDoesConstruct);
REGISTER_CPU_OPERATOR_WITH_ENGINE(JustTest, BAZ, JustTestAndDoesConstruct);
REGISTER_CUDA_OPERATOR(JustTest, JustTest);
REGISTER_CPU_OPERATOR(JustTestWithSomeOutput, JustTestWithSomeOutput);

TEST(OperatorTest, DeviceTypeRegistryWorks) {
  EXPECT_EQ(gDeviceTypeRegistry()->count(CPU), 1);
}

TEST(OperatorTest, RegistryWorks) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTest");
  unique_ptr<OperatorBase> op = CreateOperator(op_def, &ws);
  EXPECT_NE(nullptr, op.get());
  // After introducing events, CUDA operator creation has to have CUDA compiled
  // as it needs to instantiate an Event object with CUDAContext. Thus we will
  // guard this test below.
  if (HasCudaRuntime()) {
    op_def.mutable_device_option()->set_device_type(PROTO_CUDA);
    op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
  }
}

TEST(OperatorTest, RegistryWrongDevice) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTypeCPUOnly");
  op_def.mutable_device_option()->set_device_type(PROTO_CUDA);
  try {
    CreateOperator(op_def, &ws);
    LOG(FATAL) << "No exception was thrown";
  } catch (const std::exception& e) {
    LOG(INFO) << "Exception " << e.what();
  }
}

TEST(OperatorTest, ExceptionWorks) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("ThrowException");
  unique_ptr<OperatorBase> op = CreateOperator(op_def, &ws);
  // Note: we do not do ASSERT_THROW in order to print out
  // the error message for inspection.
  try {
    op->Run();
    // This should not happen - exception should throw above.
    LOG(FATAL) << "This should not happen.";
  } catch (const EnforceNotMet& err) {
    LOG(INFO) << err.msg();
  }
  try {
    op->RunAsync();
    // This should not happen - exception should throw above.
    LOG(FATAL) << "This should not happen.";
  } catch (const EnforceNotMet& err) {
    LOG(INFO) << err.msg();
  }
}

TEST(OperatorTest, FallbackIfEngineDoesNotBuild) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTest");
  op_def.set_engine("FOO");
  unique_ptr<OperatorBase> op = CreateOperator(op_def, &ws);
  EXPECT_NE(nullptr, op.get());
  EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "base");
}

TEST(OperatorTest, MultipleEngineChoices) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTest");
  op_def.set_engine("FOO,BAR");
  unique_ptr<OperatorBase> op = CreateOperator(op_def, &ws);
  EXPECT_NE(nullptr, op.get());
  EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "BAR");
}

TEST(OperatorTest, CannotUseUninitializedBlob) {
  Workspace ws;
  OperatorDef op_def;
  op_def.set_name("JustTest0");
  op_def.set_type("JustTest");
  op_def.add_input("input");
  op_def.add_output("output");
  ASSERT_THROW(CreateOperator(op_def, &ws), EnforceNotMet);
}

TEST(OperatorTest, TestParameterAccess) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_name("JustTest0");
  op_def.set_type("JustTest");
  op_def.add_input("input");
  op_def.add_output("output");
  AddArgument<float>("arg0", 0.1, &op_def);
  AddArgument<vector<int>>("arg1", vector<int>{1, 2}, &op_def);
  AddArgument<string>("arg2", "argstring", &op_def);
  EXPECT_NE(ws.CreateBlob("input"), nullptr);
  OperatorBase op(op_def, &ws);
  EXPECT_FLOAT_EQ(op.GetSingleArgument<float>("arg0", 0.0), 0.1);
  vector<int> i = op.GetRepeatedArgument<int>("arg1");
  EXPECT_EQ(i.size(), 2);
  EXPECT_EQ(i[0], 1);
  EXPECT_EQ(i[1], 2);
  EXPECT_EQ(op.GetSingleArgument<string>("arg2", "default"), "argstring");
  auto default1 = op.GetRepeatedArgument<int>("arg3", {2, 3});
  EXPECT_EQ(default1.size(), 2);
  EXPECT_EQ(default1[0], 2);
  EXPECT_EQ(default1[1], 3);
  auto default2 = op.GetRepeatedArgument<int>("arg4");
  EXPECT_EQ(default2.size(), 0);
}

TEST(OperatorTest, CannotAccessParameterWithWrongType) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_name("JustTest0");
  op_def.set_type("JustTest");
  op_def.add_input("input");
  op_def.add_output("output");
  AddArgument<float>("arg0", 0.1f, &op_def);
  EXPECT_NE(ws.CreateBlob("input"), nullptr);
  OperatorBase op(op_def, &ws);
  EXPECT_FLOAT_EQ(op.GetSingleArgument<float>("arg0", 0.0), 0.1);
  ASSERT_THROW(op.GetSingleArgument<int>("arg0", 0), EnforceNotMet);
}

#if GTEST_HAS_DEATH_TEST
TEST(OperatorDeathTest, DISABLED_CannotAccessRepeatedParameterWithWrongType) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_name("JustTest0");
  op_def.set_type("JustTest");
  op_def.add_input("input");
  op_def.add_output("output");
  AddArgument<vector<float>>("arg0", vector<float>{0.1f}, &op_def);
  EXPECT_NE(ws.CreateBlob("input"), nullptr);
  OperatorBase op(op_def, &ws);
  auto args = op.GetRepeatedArgument<float>("arg0");
  EXPECT_EQ(args.size(), 1);
  EXPECT_FLOAT_EQ(args[0], 0.1f);
  EXPECT_DEATH(op.GetRepeatedArgument<int>("arg0"),
               "Argument does not have the right field: expected ints");
}
#endif

TEST(OperatorTest, TestDefaultValue) {
  OperatorDef op_def;
  Workspace ws;
  OperatorBase op(op_def, &ws);
  EXPECT_FLOAT_EQ(op.GetSingleArgument<float>("arg-nonexisting", 0.5f), 0.5f);
}

TEST(OperatorTest, TestSetUp) {
  Workspace ws;
  OperatorDef op_def;
  op_def.set_name("JustTest0");
  op_def.set_type("JustTest");
  op_def.add_input("input");
  op_def.add_output("output");
  EXPECT_NE(nullptr, ws.CreateBlob("input"));
  unique_ptr<OperatorBase> op(CreateOperator(op_def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(ws.HasBlob("output"));
}

TEST(OperatorTest, TestSetUpInputOutputCount) {
  Workspace ws;
  OperatorDef op_def;
  op_def.set_name("JustTest0");
  op_def.set_type("JustTest");
  op_def.add_input("input");
  op_def.add_input("input2");
  op_def.add_output("output");
  EXPECT_NE(nullptr, ws.CreateBlob("input"));
  EXPECT_NE(nullptr, ws.CreateBlob("input2"));
#ifndef CAFFE2_NO_OPERATOR_SCHEMA
  // JustTest will only accept one single input.
  ASSERT_ANY_THROW(CreateOperator(op_def, &ws));
#endif

  op_def.clear_input();
  op_def.add_input("input");
  op_def.add_output("output2");
#ifndef CAFFE2_NO_OPERATOR_SCHEMA
  // JustTest will only produce one single output.
  ASSERT_ANY_THROW(CreateOperator(op_def, &ws));
#endif
}

TEST(OperatorTest, TestOutputValues) {
  NetDef net_def;
  net_def.set_name("NetForTest");
  OperatorDef op_def;
  Workspace ws;
  op_def.set_name("JustTest1");
  op_def.set_type("JustTestWithSomeOutput");
  op_def.add_output("output");
  // JustTest will only produce one single output.
  net_def.add_op()->CopyFrom(op_def);
  unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  EXPECT_TRUE(net->Run());
  EXPECT_TRUE(ws.HasBlob("output"));
  EXPECT_EQ(ws.GetBlob("output")->Get<int>(), 5);
}

NetDef GetNetDefForTest() {
  NetDef net_def;
  OperatorDef op_def;
  net_def.set_name("NetForTest");
  op_def.set_name("JustTest0");
  op_def.set_type("JustTest");
  op_def.add_input("input");
  op_def.add_output("hidden");
  net_def.add_op()->CopyFrom(op_def);
  op_def.set_name("JustTest1");
  op_def.set_input(0, "hidden");
  op_def.set_output(0, "output");
  net_def.add_op()->CopyFrom(op_def);
  return net_def;
}

TEST(NetTest, TestScaffoldingSimpleNet) {
  NetDef net_def = GetNetDefForTest();
  net_def.set_type("simple");
  Workspace ws;
  EXPECT_NE(nullptr, ws.CreateBlob("input"));
  unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  EXPECT_NE(nullptr, net.get());
  EXPECT_TRUE(ws.HasBlob("input"));
  EXPECT_TRUE(ws.HasBlob("hidden"));
  EXPECT_TRUE(ws.HasBlob("output"));
  EXPECT_TRUE(net->Run());
}

TEST(NetTest, TestScaffoldingDAGNet) {
  NetDef net_def = GetNetDefForTest();
  net_def.set_type("dag");
  net_def.set_num_workers(1);
  Workspace ws;
  EXPECT_NE(nullptr, ws.CreateBlob("input"));
  unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  EXPECT_NE(nullptr, net.get());
  EXPECT_TRUE(ws.HasBlob("input"));
  EXPECT_TRUE(ws.HasBlob("hidden"));
  EXPECT_TRUE(ws.HasBlob("output"));
  EXPECT_TRUE(net->Run());
}

class FooGradientOp : public JustTest {
 public:
  using JustTest::JustTest;
  string type() override {
    return "FooGradient";
  }
};

class FooGradientDummyEngineOp : public JustTest {
 public:
  using JustTest::JustTest;
  string type() override {
    return "FooGradientDummyEngine";
  }
};

class GetFooGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return vector<OperatorDef>{
        CreateOperatorDef(
            "FooGradient", "",
            std::vector<string>{GO(0)},
            std::vector<string>{GI(0)})};
  }
};

GRADIENT_OPERATOR_SCHEMA(FooGradient).NumInputs(1).NumOutputs(1);
REGISTER_CPU_GRADIENT_OPERATOR(FooGradient, FooGradientOp)
REGISTER_CPU_GRADIENT_OPERATOR_WITH_ENGINE(
    FooGradient,
    DUMMY_ENGINE,
    FooGradientDummyEngineOp)
REGISTER_GRADIENT(Foo, GetFooGradient);

TEST(OperatorGradientRegistryTest, GradientSimple) {
  Argument arg = MakeArgument<int>("arg", 1);
  DeviceOption option;
  option.set_device_type(PROTO_CPU);
  OperatorDef def = CreateOperatorDef(
      "Foo", "", std::vector<string>{"in"}, std::vector<string>{"out"},
      std::vector<Argument>{arg}, option, "DUMMY_ENGINE");
  vector<GradientWrapper> g_output(1);
  g_output[0].dense_ = "out_grad";
  GradientOpsMeta meta = GetGradientForOp(def, g_output);
  // Check the names, input and output.
  EXPECT_EQ(meta.ops_.size(), 1);
  const OperatorDef& grad_op_def = meta.ops_[0];
  EXPECT_EQ(grad_op_def.type(), "FooGradient");
  EXPECT_EQ(grad_op_def.name(), "");
  EXPECT_EQ(grad_op_def.input_size(), 1);
  EXPECT_EQ(grad_op_def.output_size(), 1);
  EXPECT_EQ(grad_op_def.input(0), "out_grad");
  EXPECT_EQ(grad_op_def.output(0), "in_grad");
  // Checks the engine, device option and arguments.
  EXPECT_EQ(grad_op_def.engine(), "DUMMY_ENGINE");
  EXPECT_EQ(grad_op_def.device_option().device_type(), PROTO_CPU);
  EXPECT_EQ(grad_op_def.arg_size(), 1);
  EXPECT_EQ(
      grad_op_def.arg(0).SerializeAsString(),
      MakeArgument<int>("arg", 1).SerializeAsString());
  // Checks the gradient name for input.
  EXPECT_EQ(meta.g_input_.size(), 1);
  EXPECT_TRUE(meta.g_input_[0].IsDense());
  EXPECT_EQ(meta.g_input_[0].dense_, "in_grad");

  Workspace ws;
  EXPECT_NE(ws.CreateBlob("out_grad"), nullptr);
  unique_ptr<OperatorBase> grad_op = CreateOperator(grad_op_def, &ws);
  EXPECT_NE(nullptr, grad_op.get());
  EXPECT_EQ(
      static_cast<JustTest*>(grad_op.get())->type(), "FooGradientDummyEngine");
}

TEST(EnginePrefTest, PerOpEnginePref) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTest");

  SetPerOpEnginePref({{CPU, {{"JustTest", {"BAR"}}}}});
  {
    const auto op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
    EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "BAR");
  }
  // clear
  SetPerOpEnginePref({});

  // Invalid operator type
  ASSERT_THROW(
      SetPerOpEnginePref({{CPU, {{"NO_EXIST", {"BAR"}}}}}), EnforceNotMet);
}

TEST(EnginePrefTest, GlobalEnginePref) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTest");

  SetGlobalEnginePref({{CPU, {"FOO", "BAR"}}});
  {
    const auto op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
    EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "BAR");
  }
  // clear
  SetGlobalEnginePref({});

  SetGlobalEnginePref({{CPU, {"FOO"}}});
  {
    const auto op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
    EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "base");
  }
  // clear
  SetGlobalEnginePref({});

  // Invalid device type
  // This check is no longer necessary with the enum class
  // ASSERT_THROW(SetGlobalEnginePref({{8888, {"FOO"}}}), EnforceNotMet);
}

TEST(EnginePrefTest, GlobalEnginePrefAndPerOpEnginePref) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTest");

  SetPerOpEnginePref({{CPU, {{"JustTest", {"BAR"}}}}});
  SetGlobalEnginePref({{CPU, {"BAZ"}}});
  {
    const auto op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
    // per op pref takes precedence
    EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "BAR");
  }
  // clear
  SetPerOpEnginePref({});
  SetGlobalEnginePref({});
}

TEST(EnginePrefTest, GlobalEnginePrefAndPerOpEnginePrefAndOpDef) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTest");
  op_def.set_engine("BAR");

  SetPerOpEnginePref({{CPU, {{"JustTest", {"BAZ"}}}}});
  SetGlobalEnginePref({{CPU, {"BAZ"}}});
  {
    const auto op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
    // operator_def takes precedence
    EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "BAR");
  }
  // clear
  SetPerOpEnginePref({});
  SetGlobalEnginePref({});
}

TEST(EnginePrefTest, SetOpEnginePref) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTest");

  SetPerOpEnginePref({{CPU, {{"JustTest", {"BAZ"}}}}});
  SetOpEnginePref("JustTest", {{CPU, {"BAR"}}});
  {
    const auto op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
    // operator_def takes precedence
    EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "BAR");
  }
  // clear
  SetPerOpEnginePref({});
  SetGlobalEnginePref({});
}

TEST(EnginePrefTest, SetDefaultEngine) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTest");

  SetPerOpEnginePref({{CPU, {{"JustTest", {"DEFAULT"}}}}});
  SetGlobalEnginePref({{CPU, {"BAR"}}});
  {
    const auto op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
    // operator_def takes precedence
    EXPECT_EQ(static_cast<JustTest*>(op.get())->type(), "base");
  }
  // clear
  SetPerOpEnginePref({});
  SetGlobalEnginePref({});
}

class JustTestWithRequiredArg : public JustTest {
 public:
  using JustTest::JustTest;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  string type() override {
    return "JustTestWithRequiredArg";
  }
};

REGISTER_CPU_OPERATOR(JustTestWithRequiredArg, JustTestWithRequiredArg);
OPERATOR_SCHEMA(JustTestWithRequiredArg)
    .NumInputs(0, 1)
    .NumOutputs(0, 1)
    .Arg("test_arg", "this arg is required", true);

TEST(RequiredArg, Basic) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTestWithRequiredArg");

  {
    try {
      CreateOperator(op_def, &ws);
      LOG(FATAL) << "No exception was thrown";
    } catch (const std::exception& e) {
      LOG(INFO) << "Exception thrown (expected): " << e.what();
    }
  }

  {
    op_def.add_arg()->CopyFrom(MakeArgument("test_arg", 1));
    const auto op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
    EXPECT_EQ(
        static_cast<JustTest*>(op.get())->type(), "JustTestWithRequiredArg");
  }
}

class JustTestWithStandardIsTestArg : public JustTest {
 public:
  using JustTest::JustTest;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  string type() override {
    return "JustTestWithStandardIsTestArg";
  }
};

REGISTER_CPU_OPERATOR(
    JustTestWithStandardIsTestArg,
    JustTestWithStandardIsTestArg);
OPERATOR_SCHEMA(JustTestWithStandardIsTestArg)
    .NumInputs(0, 1)
    .NumOutputs(0, 1)
    .ArgIsTest("this is_test arg is required");

TEST(IsTestArg, standard) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTestWithStandardIsTestArg");

  {
    try {
      CreateOperator(op_def, &ws);
      LOG(FATAL) << "No exception was thrown";
    } catch (const std::exception& e) {
      LOG(INFO) << "Exception thrown (expected): " << e.what();
    }
  }

  {
    op_def.add_arg()->CopyFrom(MakeArgument(OpSchema::Arg_IsTest, 1));
    const auto op = CreateOperator(op_def, &ws);
    EXPECT_NE(nullptr, op.get());
    EXPECT_EQ(
        static_cast<JustTest*>(op.get())->type(),
        "JustTestWithStandardIsTestArg");
  }
}

class JustTestWithNonStandardIsTestArg : public JustTest {
 public:
  using JustTest::JustTest;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  string type() override {
    return "JustTestWithNonStandardIsTestArg";
  }
};

REGISTER_CPU_OPERATOR(
    JustTestWithNonStandardIsTestArg,
    JustTestWithNonStandardIsTestArg);
OPERATOR_SCHEMA(JustTestWithNonStandardIsTestArg)
    .NumInputs(0, 1)
    .NumOutputs(0, 1)
    .Arg(OpSchema::Arg_IsTest, "this is_test arg is not required");

TEST(IsTestArg, non_standard) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTestWithNonStandardIsTestArg");

  const auto op = CreateOperator(op_def, &ws);
  EXPECT_NE(nullptr, op.get());
  EXPECT_EQ(
      static_cast<JustTest*>(op.get())->type(),
      "JustTestWithNonStandardIsTestArg");
}

}  // namespace caffe2
