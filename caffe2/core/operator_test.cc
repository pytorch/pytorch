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
  INPUT_OUTPUT_STATS(0, 1, 0, 1);
};
REGISTER_CPU_OPERATOR(JustTest, JustTest);
REGISTER_CUDA_OPERATOR(JustTest, JustTest);


TEST(OperatorTest, RegistryWorks) {
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("JustTest");
  EXPECT_NE(nullptr, CreateOperator(op_def, &ws));
  op_def.mutable_device_option()->set_device_type(CUDA);
  EXPECT_NE(nullptr, CreateOperator(op_def, &ws));

  CPUOperatorRegistry()->TEST_PrintRegisteredNames();
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
  EXPECT_TRUE(op.Verify());
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
  EXPECT_TRUE(op.Verify());
  EXPECT_FLOAT_EQ(op.GetSingleArgument<float>("arg0", 0.0), 0.1);
  EXPECT_DEATH(op.GetSingleArgument<int>("arg0", 0),
               "Argument does not have the right field: expected i");
}

TEST(OperatorDeathTest, CannotAccessRepeatedParameterWithWrongType) {
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
  EXPECT_TRUE(op.Verify());
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
  EXPECT_TRUE(op->Verify());
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
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(ws.HasBlob("output"));
  // Because JustTest will only accept one single input, this will return false.
  EXPECT_FALSE(op->Verify());

  op_def.clear_input();
  op_def.add_input("input");
  op_def.add_output("output2");
  op.reset(CreateOperator(op_def, &ws));
  EXPECT_NE(nullptr, op.get());
  // Because JustTest will only produce one single output, this will return
  // false.
  EXPECT_FALSE(op->Verify());
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

TEST(NetTest, TestScaffoldingParallelNet) {
  NetDef net_def = GetNetDefForTest();
  net_def.set_net_type("parallel");
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

}  // namespace caffe2


