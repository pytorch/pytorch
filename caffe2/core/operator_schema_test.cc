#include "caffe2/core/operator_schema.h"
#include "caffe2/core/logging.h"
#include "caffe2/utils/proto_utils.h"

#include "gtest/gtest.h"

namespace caffe2 {

OPERATOR_SCHEMA(OpSchemaTestOp)
  .NumInputs(1).NumOutputs(1);

TEST(OperatorSchemaTest, BasicSchema) {
  const OpSchema* schema = OpSchemaRegistry::Schema("OpSchemaTestOp");
  EXPECT_TRUE(schema != nullptr);
  OperatorDef def1 = CreateOperatorDef(
      "OpSchemaTestOp", "",
      vector<string>{"in"}, vector<string>{"out"});
  EXPECT_TRUE(schema->Verify(def1));
  OperatorDef def2 = CreateOperatorDef(
      "OpSchemaTestOp", "",
      vector<string>{"in1", "in2"}, vector<string>{"out"});
  EXPECT_FALSE(schema->Verify(def2));
  OperatorDef def3 = CreateOperatorDef(
      "OpSchemaTestOp", "",
      vector<string>{"in"}, vector<string>{"out1", "out2"});
  EXPECT_FALSE(schema->Verify(def3));
}

OPERATOR_SCHEMA(OpSchemaSpecifiedInputOutputOp)
  .NumInputs({2, 4}).NumOutputs({1, 3});

TEST(OperatorSchemaTest, SpecifiedInputOutput) {
  const OpSchema* schema
      = OpSchemaRegistry::Schema("OpSchemaSpecifiedInputOutputOp");
  EXPECT_TRUE(schema != nullptr);
  OperatorDef def1 = CreateOperatorDef(
      "OpSchemaSpecifiedInputOutputOp", "",
      vector<string>{"in"}, vector<string>{"out"});
  EXPECT_FALSE(schema->Verify(def1));
  OperatorDef def2 = CreateOperatorDef(
      "OpSchemaSpecifiedInputOutputOp", "",
      vector<string>{"in1", "in2"}, vector<string>{"out"});
  EXPECT_TRUE(schema->Verify(def2));
  OperatorDef def3 = CreateOperatorDef(
      "OpSchemaSpecifiedInputOutputOp", "",
      vector<string>{"in1", "in2"}, vector<string>{"out1", "out2"});
  EXPECT_FALSE(schema->Verify(def3));
}

OPERATOR_SCHEMA(OpSchemaSameInputOutputOp)
    .SameNumberOfOutput();

TEST(OperatorSchemaTest, SameInputOutput) {
  const OpSchema* schema =
      OpSchemaRegistry::Schema("OpSchemaSameInputOutputOp");
  OperatorDef def1 = CreateOperatorDef(
      "OpSchemaSameInputOutputOp", "",
      vector<string>{"in"}, vector<string>{"out"});
  EXPECT_TRUE(schema->Verify(def1));
  OperatorDef def2 = CreateOperatorDef(
      "OpSchemaSameInputOutputOp", "",
      vector<string>{"in1", "in2"}, vector<string>{"out1", "out2"});
  EXPECT_TRUE(schema->Verify(def2));
  OperatorDef def3 = CreateOperatorDef(
      "OpSchemaSameInputOutputOp", "",
      vector<string>{"in1", "in2"}, vector<string>{"out1", "out2", "out3"});
  EXPECT_FALSE(schema->Verify(def3));
}

OPERATOR_SCHEMA(OpSchemaCalculateOutputOp)
    .NumInputs(1, 5).NumOutputs(2, 6)
    .OutputCalculator([](int n) { return n + 1; });

TEST(OperatorSchemaTest, CalculateOutput) {
  const OpSchema* schema =
      OpSchemaRegistry::Schema("OpSchemaCalculateOutputOp");
  OperatorDef def1 = CreateOperatorDef(
      "OpSchemaCalculateOutputOp", "",
      vector<string>{"in"}, vector<string>{"out"});
  EXPECT_FALSE(schema->Verify(def1));
  OperatorDef def2 = CreateOperatorDef(
      "OpSchemaCalculateOutputOp", "",
      vector<string>{"in1", "in2"}, vector<string>{"out1", "out2"});
  EXPECT_FALSE(schema->Verify(def2));
  OperatorDef def3 = CreateOperatorDef(
      "OpSchemaCalculateOutputOp", "",
      vector<string>{"in1", "in2"}, vector<string>{"out1", "out2", "out3"});
  EXPECT_TRUE(schema->Verify(def3));
}

OPERATOR_SCHEMA(OpSchemaInplace)
    .NumInputs(2).NumOutputs(2)
    .AllowInplace({{0, 0}})
    .EnforceInplace({{1, 1}});

TEST(OperatorSchemaTest, Inplace) {
  const OpSchema* schema =
      OpSchemaRegistry::Schema("OpSchemaInplace");
  OperatorDef def1 = CreateOperatorDef(
      "OpSchemaInplace", "",
      vector<string>{"in1", "in2"}, vector<string>{"out1", "in2"});
  EXPECT_TRUE(schema->Verify(def1));
  OperatorDef def2 = CreateOperatorDef(
      "OpSchemaInplace", "",
      vector<string>{"in1", "in2"}, vector<string>{"in1", "in2"});
  EXPECT_TRUE(schema->Verify(def2));
  OperatorDef def3 = CreateOperatorDef(
      "OpSchemaInplace", "",
      vector<string>{"in1", "in2"}, vector<string>{"in1", "out2"});
  EXPECT_FALSE(schema->Verify(def3));
  OperatorDef def4 = CreateOperatorDef(
      "OpSchemaInplace", "",
      vector<string>{"in1", "in2"}, vector<string>{"out1", "out2"});
  EXPECT_FALSE(schema->Verify(def4));
}

}  // namespace caffe2
