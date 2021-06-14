#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/utils/proto_utils.h"

#include <gtest/gtest.h>

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(OpSchemaTestOp)
  .NumInputs(1).NumOutputs(1)
  .SetDoc(R"DOC(Test Documentation)DOC")
  .Input(0, "in0", "dummy input.")
  .Output(0, "out0", "dummy output.");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorSchemaTest, BasicSchema) {
  const OpSchema* schema = OpSchemaRegistry::Schema("OpSchemaTestOp");
#ifdef CAFFE2_NO_OPERATOR_SCHEMA
  EXPECT_TRUE(schema == nullptr);
  return;
#endif
  EXPECT_TRUE(schema != nullptr);
  EXPECT_TRUE(schema->doc() != nullptr);
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(OpSchemaSpecifiedInputOutputOp)
  .NumInputs({2, 4}).NumOutputs({1, 3});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorSchemaTest, SpecifiedInputOutput) {
  const OpSchema* schema
      = OpSchemaRegistry::Schema("OpSchemaSpecifiedInputOutputOp");
#ifdef CAFFE2_NO_OPERATOR_SCHEMA
  EXPECT_TRUE(schema == nullptr);
  return;
#endif
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(OpSchemaInputOutputRelationOp)
    .NumInputsOutputs([](int in, int out) {
      return out == in || out == in * 2;
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorSchemaTest, InputOutputRelation) {
  const OpSchema* schema
      = OpSchemaRegistry::Schema("OpSchemaInputOutputRelationOp");
#ifdef CAFFE2_NO_OPERATOR_SCHEMA
  EXPECT_TRUE(schema == nullptr);
  return;
#endif
  EXPECT_TRUE(schema != nullptr);
  OperatorDef def1 = CreateOperatorDef(
      "OpSchemaInputOutputRelationOp", "",
      vector<string>{"in"}, vector<string>{"out"});
  EXPECT_TRUE(schema->Verify(def1));
  OperatorDef def2 = CreateOperatorDef(
      "OpSchemaInputOutputRelationOp", "",
      vector<string>{"in"}, vector<string>{"out1", "out2"});
  EXPECT_TRUE(schema->Verify(def2));
  OperatorDef def3 = CreateOperatorDef(
      "OpSchemaInputOutputRelationOp", "",
      vector<string>{"in1", "in2", "in3"}, vector<string>{"out1", "out2"});
  EXPECT_FALSE(schema->Verify(def3));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(OpSchemaSameInputOutputOp)
    .SameNumberOfOutput();

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorSchemaTest, SameInputOutput) {
  const OpSchema* schema =
      OpSchemaRegistry::Schema("OpSchemaSameInputOutputOp");
#ifdef CAFFE2_NO_OPERATOR_SCHEMA
  EXPECT_TRUE(schema == nullptr);
  return;
#endif
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(OpSchemaCalculateOutputOp)
    .NumInputs(1, 5).NumOutputs(2, 6)
    .OutputCalculator([](int n) { return n + 1; });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorSchemaTest, CalculateOutput) {
  const OpSchema* schema =
      OpSchemaRegistry::Schema("OpSchemaCalculateOutputOp");
#ifdef CAFFE2_NO_OPERATOR_SCHEMA
  EXPECT_TRUE(schema == nullptr);
  return;
#endif
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(OpSchemaInplace)
    .NumInputs(2).NumOutputs(2)
    .AllowInplace({{0, 0}})
    .EnforceInplace({{1, 1}});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorSchemaTest, Inplace) {
  const OpSchema* schema =
      OpSchemaRegistry::Schema("OpSchemaInplace");
#ifdef CAFFE2_NO_OPERATOR_SCHEMA
  EXPECT_TRUE(schema == nullptr);
  return;
#endif
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(OpSchemaSameInputOutputTensorInference).IdenticalTypeAndShape();

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorSchemaTest, TensorInferenceIdentical) {
  const OpSchema* schema =
      OpSchemaRegistry::Schema("OpSchemaSameInputOutputTensorInference");
#ifdef CAFFE2_NO_OPERATOR_SCHEMA
  EXPECT_TRUE(schema == nullptr);
  return;
#endif
  OperatorDef def = CreateOperatorDef(
      "OpSchemaSameInputOutputTensorInference",
      "",
      vector<string>{"in"},
      vector<string>{"out"});
  vector<TensorShape> shapes(1);
  shapes[0].set_data_type(TensorProto::FLOAT);
  shapes[0].add_dims(1);
  shapes[0].add_dims(2);
  shapes[0].add_dims(3);
  vector<TensorShape> out = schema->InferTensor(def, shapes);
  EXPECT_EQ(out.size(), 1);
  EXPECT_EQ(out[0].SerializeAsString(), shapes[0].SerializeAsString());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(OpSchemaArbitraryTensorInference)
    .TensorInferenceFunction(
        [](const OperatorDef&, const vector<TensorShape>&) {
          vector<TensorShape> shapes(1);
          shapes[0].set_data_type(TensorProto::FLOAT);
          shapes[0].add_dims(1701);
          return shapes;
        });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorSchemaTest, TensorInferenceArbitrary) {
  const OpSchema* schema =
      OpSchemaRegistry::Schema("OpSchemaArbitraryTensorInference");
#ifdef CAFFE2_NO_OPERATOR_SCHEMA
  EXPECT_TRUE(schema == nullptr);
  return;
#endif
  OperatorDef def = CreateOperatorDef(
      "OpSchemaArbitraryTensorInference",
      "",
      vector<string>{"in"},
      vector<string>{"out"});
  vector<TensorShape> out = schema->InferTensor(def, vector<TensorShape>());
  EXPECT_EQ(out.size(), 1);
  EXPECT_EQ(out[0].data_type(), TensorProto::FLOAT);
  EXPECT_EQ(out[0].dims_size(), 1);
  EXPECT_EQ(out[0].dims(0), 1701);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorSchemaTest, TestCastSchema) {
  // This tests a use case of the schema: the Cast op takes in the def and
  // deduces the
  // schema from the "to" argument.
  const OpSchema* schema = OpSchemaRegistry::Schema("Cast");
#ifdef CAFFE2_NO_OPERATOR_SCHEMA
  EXPECT_TRUE(schema == nullptr);
  return;
#endif
  if (!schema) {
    // Compiled without the Cast op.
    return;
  }
  OperatorDef def = CreateOperatorDef(
      "Cast",
      "",
      vector<string>{"in"},
      vector<string>{"out"},
      vector<Argument>{MakeArgument<int64_t>("to", TensorProto::UINT8)});
  vector<TensorShape> out = schema->InferTensor(def, vector<TensorShape>(1));
  EXPECT_EQ(out.size(), 1);
  // Data type should be inferred.
  EXPECT_EQ(out[0].data_type(), TensorProto::UINT8);
  // Dim should not be set (same as input);
  EXPECT_EQ(out[0].dims_size(), 0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(OpSchemaCostInference)
    .NumInputs(2)
    .NumOutputs(2)
    .CostInferenceFunction([](const OperatorDef& /*def*/,
                              const vector<TensorShape>& inputs) {
      struct OpSchema::Cost c;
      c.flops = 2 * inputs[0].dims(0) * inputs[0].dims(1) * inputs[1].dims(1);
      return c;
    });

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(OperatorSchemaTest, TestCostInference) {
  const OpSchema* schema = OpSchemaRegistry::Schema("OpSchemaCostInference");
#ifdef CAFFE2_NO_OPERATOR_SCHEMA
  EXPECT_TRUE(schema == nullptr);
  return;
#endif
  if (!schema) {
    return;
  }
  OperatorDef def = CreateOperatorDef(
      "OpSchemaCostInference", "", vector<string>{"in"}, vector<string>{"out"});
  vector<TensorShape> shapes(2);
  shapes[0].set_data_type(TensorProto::FLOAT);
  shapes[0].add_dims(10);
  shapes[0].add_dims(10);
  shapes[1].set_data_type(TensorProto::FLOAT);
  shapes[1].add_dims(10);
  shapes[1].add_dims(10);
  EXPECT_EQ(2000, schema->InferCost(def, shapes).flops);
}

}  // namespace caffe2
