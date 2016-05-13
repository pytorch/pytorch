#include <iostream>

#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "gtest/gtest.h"

namespace caffe2 {

class JustTest : public OperatorBase {
 public:
  explicit JustTest(const OperatorDef& op_def, Workspace* ws)
      : OperatorBase(op_def, ws) {}
  bool Run() override { return true; }
};

OPERATOR_SCHEMA(JustTest)
    .NumInputs(0, 1).NumOutputs(0, 1);

REGISTER_CPU_OPERATOR(JustTest, JustTest);
REGISTER_CUDA_OPERATOR(JustTest, JustTest);


TEST(OperatorTest, RegistryWorks) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTest");
  unique_ptr<OperatorBase> op(CreateOperator(op_def, &ws));
  EXPECT_NE(nullptr, op.get());
  op_def.mutable_device_option()->set_device_type(CUDA);
  op.reset(CreateOperator(op_def, &ws));
  EXPECT_NE(nullptr, op.get());
}

TEST(OperatorDeathTest, CannotUseUninitializedBlob) {
  Workspace ws;
  OperatorDef op_def;
  op_def.set_name("JustTest0");
  op_def.set_type("JustTest");
  op_def.add_input("input");
  op_def.add_output("output");
  EXPECT_DEATH(CreateOperator(op_def, &ws), "Check failed");
}

TEST(OperatorTest, TestParameterAccess) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_name("JustTest0");
  op_def.set_type("JustTest");
  op_def.add_input("input");
  op_def.add_output("output");
  {
    Argument* arg = op_def.add_arg();
    arg->set_name("arg0");
    arg->set_f(0.1);
  }
  {
    Argument* arg = op_def.add_arg();
    arg->set_name("arg1");
    arg->add_ints(1);
    arg->add_ints(2);
  }
  {
    Argument* arg = op_def.add_arg();
    arg->set_name("arg2");
    arg->set_s("argstring");
  }
  EXPECT_NE(ws.CreateBlob("input"), nullptr);
  OperatorBase op(op_def, &ws);
  EXPECT_FLOAT_EQ(op.GetSingleArgument<float>("arg0", 0.0), 0.1);
  vector<int> i = op.GetRepeatedArgument<int>("arg1");
  EXPECT_EQ(i.size(), 2);
  EXPECT_EQ(i[0], 1);
  EXPECT_EQ(i[1], 2);
  EXPECT_EQ(op.GetSingleArgument<string>("arg2", "default"), "argstring");
}


TEST(OperatorDeathTest, CannotAccessParameterWithWrongType) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_name("JustTest0");
  op_def.set_type("JustTest");
  op_def.add_input("input");
  op_def.add_output("output");
  {
    Argument* arg = op_def.add_arg();
    arg->set_name("arg0");
    arg->set_f(0.1);
  }
  EXPECT_NE(ws.CreateBlob("input"), nullptr);
  OperatorBase op(op_def, &ws);
  EXPECT_FLOAT_EQ(op.GetSingleArgument<float>("arg0", 0.0), 0.1);
  EXPECT_DEATH(op.GetSingleArgument<int>("arg0", 0),
               "Argument arg0 does not have the right field: expected field i");
}

TEST(OperatorDeathTest, DISABLED_CannotAccessRepeatedParameterWithWrongType) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_name("JustTest0");
  op_def.set_type("JustTest");
  op_def.add_input("input");
  op_def.add_output("output");
  {
    Argument* arg = op_def.add_arg();
    arg->set_name("arg0");
    arg->add_floats(0.1);
  }
  EXPECT_NE(ws.CreateBlob("input"), nullptr);
  OperatorBase op(op_def, &ws);
  auto args = op.GetRepeatedArgument<float>("arg0");
  EXPECT_EQ(args.size(), 1);
  EXPECT_FLOAT_EQ(args[0], 0.1);
  EXPECT_DEATH(op.GetRepeatedArgument<int>("arg0"),
               "Argument does not have the right field: expected ints");
}

TEST(OperatorTest, TestDefaultValue) {
  OperatorDef op_def;
  Workspace ws;
  OperatorBase op(op_def, &ws);
  EXPECT_FLOAT_EQ(
      op.GetSingleArgument<float>("arg-nonexisting", 0.5), 0.5);
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
  unique_ptr<OperatorBase> op(CreateOperator(op_def, &ws));
  // JustTest will only accept one single input.
  EXPECT_EQ(nullptr, op.get());

  op_def.clear_input();
  op_def.add_input("input");
  op_def.add_output("output2");
  op.reset(CreateOperator(op_def, &ws));
  // JustTest will only produce one single output.
  EXPECT_EQ(nullptr, op.get());
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
  net_def.set_net_type("simple");
  Workspace ws;
  EXPECT_NE(nullptr, ws.CreateBlob("input"));
  unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  EXPECT_NE(nullptr, net.get());
  EXPECT_TRUE(net->Verify());
  EXPECT_TRUE(ws.HasBlob("input"));
  EXPECT_TRUE(ws.HasBlob("hidden"));
  EXPECT_TRUE(ws.HasBlob("output"));
  EXPECT_TRUE(net->Run());
}

TEST(NetTest, TestScaffoldingDAGNet) {
  NetDef net_def = GetNetDefForTest();
  net_def.set_net_type("dag");
  net_def.set_num_workers(1);
  Workspace ws;
  EXPECT_NE(nullptr, ws.CreateBlob("input"));
  unique_ptr<NetBase> net(CreateNet(net_def, &ws));
  EXPECT_NE(nullptr, net.get());
  EXPECT_TRUE(net->Verify());
  EXPECT_TRUE(ws.HasBlob("input"));
  EXPECT_TRUE(ws.HasBlob("hidden"));
  EXPECT_TRUE(ws.HasBlob("output"));
  EXPECT_TRUE(net->Run());
}

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

REGISTER_GRADIENT(Foo, GetFooGradient);

TEST(OperatorGradientRegistryTest, GradientSimple) {
  Argument arg = MakeArgument<int>("arg", 1);
  DeviceOption option;
  option.set_device_type(CPU);
  OperatorDef def = CreateOperatorDef(
      "Foo", "", std::vector<string>{"in"}, std::vector<string>{"out"},
      std::vector<Argument>{arg}, option, "DUMMY_ENGINE");
  vector<GradientWrapper> g_output(1);
  g_output[0].dense_ = "out_grad";
  GradientOpsMeta meta = GetGradientForOp(def, g_output);
  // Check the names, input and output.
  EXPECT_EQ(meta.ops_.size(), 1);
  const OperatorDef& grad_op = meta.ops_[0];
  EXPECT_EQ(grad_op.type(), "FooGradient");
  EXPECT_EQ(grad_op.name(), "");
  EXPECT_EQ(grad_op.input_size(), 1);
  EXPECT_EQ(grad_op.output_size(), 1);
  EXPECT_EQ(grad_op.input(0), "out_grad");
  EXPECT_EQ(grad_op.output(0), "in_grad");
  // Checks the engine, device option and arguments.
  EXPECT_EQ(grad_op.engine(), "DUMMY_ENGINE");
  EXPECT_EQ(grad_op.device_option().device_type(), CPU);
  EXPECT_EQ(grad_op.arg_size(), 1);
  EXPECT_EQ(grad_op.arg(0).SerializeAsString(),
            MakeArgument<int>("arg", 1).SerializeAsString());
  // Checks the gradient name for input.
  EXPECT_EQ(meta.g_input_.size(), 1);
  EXPECT_TRUE(meta.g_input_[0].IsDense());
  EXPECT_EQ(meta.g_input_[0].dense_, "in_grad");
}



}  // namespace caffe2


