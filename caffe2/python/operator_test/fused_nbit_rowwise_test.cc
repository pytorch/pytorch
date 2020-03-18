#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/utils/proto_utils.h"

#include <gtest/gtest.h>

namespace caffe2 {

TEST(OperatorSchemaTest, TensorInferenceNbit) {
  for (int bit_rate : {2, 4}) {
    const OpSchema* schema = OpSchemaRegistry::Schema(
        "FloatToFused" + std::to_string(bit_rate) + "BitRowwiseQuantized");
    EXPECT_TRUE(schema != nullptr);

    OperatorDef def = CreateOperatorDef(
        "FloatToFused" + std::to_string(bit_rate) + "BitRowwiseQuantized",
        "",
        vector<string>{"in"},
        vector<string>{"out"});
    vector<TensorShape> in_shapes(1);
    in_shapes[0].set_data_type(TensorProto::FLOAT);
    in_shapes[0].add_dims(20000);
    in_shapes[0].add_dims(64);

    vector<TensorShape> out = schema->InferTensor(def, in_shapes);
    EXPECT_EQ(out.size(), 1);
    EXPECT_EQ(out[0].data_type(), TensorProto::UINT8);
    EXPECT_EQ(out[0].dims_size(), 2);
    EXPECT_EQ(out[0].dims(0), 20000);
    EXPECT_EQ(out[0].dims(1), 64 / (8 / bit_rate) + 4);
  }
}

TEST(OperatorSchemaTest, TensorInferenceNbitHalf) {
  for (int bit_rate : {2, 4}) {
    const OpSchema* schema = OpSchemaRegistry::Schema(
        "HalfToFused" + std::to_string(bit_rate) + "BitRowwiseQuantized");
    EXPECT_TRUE(schema != nullptr);

    OperatorDef def = CreateOperatorDef(
        "HalfToFused" + std::to_string(bit_rate) + "BitRowwiseQuantized",
        "",
        vector<string>{"in"},
        vector<string>{"out"});
    vector<TensorShape> in_shapes(1);
    in_shapes[0].set_data_type(TensorProto::FLOAT16);
    in_shapes[0].add_dims(20000);
    in_shapes[0].add_dims(64);

    vector<TensorShape> out = schema->InferTensor(def, in_shapes);
    EXPECT_EQ(out.size(), 1);
    EXPECT_EQ(out[0].data_type(), TensorProto::UINT8);
    EXPECT_EQ(out[0].dims_size(), 2);
    EXPECT_EQ(out[0].dims(0), 20000);
    EXPECT_EQ(out[0].dims(1), 64 / (8 / bit_rate) + 4);
  }
}

TEST(OperatorSchemaTest, TensorInferenceNbitBack) {
  for (int bit_rate : {2, 4}) {
    const OpSchema* schema = OpSchemaRegistry::Schema(
        "Fused" + std::to_string(bit_rate) + "BitRowwiseQuantizedToFloat");
    EXPECT_TRUE(schema != nullptr);

    OperatorDef def = CreateOperatorDef(
        "Fused" + std::to_string(bit_rate) + "BitRowwiseQuantizedToFloat",
        "",
        vector<string>{"in"},
        vector<string>{"out"});
    vector<TensorShape> in_shapes(1);
    in_shapes[0].set_data_type(TensorProto::UINT8);
    in_shapes[0].add_dims(20000);
    in_shapes[0].add_dims(36);

    vector<TensorShape> out = schema->InferTensor(def, in_shapes);
    EXPECT_EQ(out.size(), 1);
    EXPECT_EQ(out[0].data_type(), TensorProto::FLOAT);
    EXPECT_EQ(out[0].dims_size(), 2);
    EXPECT_EQ(out[0].dims(0), 20000);
    EXPECT_EQ(out[0].dims(1), (36 - 4) * (8 / bit_rate));
  }
}

TEST(OperatorSchemaTest, TensorInferenceNbitHalfBack) {
  for (int bit_rate : {2, 4}) {
    const OpSchema* schema = OpSchemaRegistry::Schema(
        "Fused" + std::to_string(bit_rate) + "BitRowwiseQuantizedToHalf");
    EXPECT_TRUE(schema != nullptr);

    OperatorDef def = CreateOperatorDef(
        "Fused" + std::to_string(bit_rate) + "BitRowwiseQuantizedToHalf",
        "",
        vector<string>{"in"},
        vector<string>{"out"});
    vector<TensorShape> in_shapes(1);
    in_shapes[0].set_data_type(TensorProto::UINT8);
    in_shapes[0].add_dims(20000);
    in_shapes[0].add_dims(36);

    vector<TensorShape> out = schema->InferTensor(def, in_shapes);
    EXPECT_EQ(out.size(), 1);
    EXPECT_EQ(out[0].data_type(), TensorProto::FLOAT16);
    EXPECT_EQ(out[0].dims_size(), 2);
    EXPECT_EQ(out[0].dims(0), 20000);
    EXPECT_EQ(out[0].dims(1), (36 - 4) * (8 / bit_rate));
  }
}

} // namespace caffe2
