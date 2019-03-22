#include <gtest/gtest.h>
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/opt/bound_shape_inferencer.h"
#include "caffe2/utils/proto_utils.h"

using namespace caffe2;
namespace {

ShapeInfo makeTensorInfo(
    ShapeInfo::DimType t,
    const std::vector<int64_t>& dims,
    TensorProto::DataType dtype = TensorProto_DataType_FLOAT) {
  ShapeInfo info;
  info.dim_type = t;
  TensorShape& shape = info.shape;
  for (const auto d : dims) {
    shape.add_dims(d);
  }
  shape.set_data_type(dtype);
  return info;
}

void verifyShapeInfo(
    const ShapeInfoMap& info,
    const std::string& name,
    ShapeInfo::DimType t,
    const std::vector<int64_t>& dims,
    TensorProto::DataType dtype = TensorProto_DataType_FLOAT) {
  LOG(INFO) << "Checking " << name;
  const auto it = info.find(name);
  ASSERT_TRUE(it != info.end());
  const auto& shape_info = it->second;
  EXPECT_EQ(shape_info.dim_type, t);
  const auto& shape = shape_info.shape;
  ASSERT_EQ(shape.dims_size(), dims.size());
  for (int i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(shape.dims(i), dims[i]);
  }
  EXPECT_EQ(shape.data_type(), dtype);
}

} // namespace

TEST(BoundShapeInference, SparseLengthsSum) {
  NetDef net;
  net.add_op()->CopyFrom(CreateOperatorDef(
      "SparseLengthsSum", "", {"Weights", "Data", "Lengths"}, {"Out"}, {}));
  ShapeInfoMap shape_map;
  shape_map.emplace(
      "Weights", makeTensorInfo(ShapeInfo::DimType::CONSTANT, {1000, 16}));
  BoundShapeSpec spec(20, 1000);
  BoundShapeInferencer eng(spec);
  eng.InferBoundShapeAndType(net, shape_map);
  const auto& out_shape = eng.shape_info();
  verifyShapeInfo(
      out_shape, "Weights", ShapeInfo::DimType::CONSTANT, {1000, 16});
  verifyShapeInfo(
      out_shape,
      "Data",
      ShapeInfo::DimType::SEQ,
      {spec.max_seq_size},
      TensorProto_DataType_INT64);
  verifyShapeInfo(
      out_shape,
      "Lengths",
      ShapeInfo::DimType::BATCH,
      {spec.max_batch_size},
      TensorProto_DataType_INT32);
  verifyShapeInfo(
      out_shape, "Out", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 16});
}

TEST(BoundShapeInference, SparseLengthsSumFused8BitRowwise) {
  NetDef net;
  net.add_op()->CopyFrom(CreateOperatorDef(
      "SparseLengthsSumFused8BitRowwise",
      "",
      {"Weights", "Data", "Lengths"},
      {"Out"},
      {}));
  ShapeInfoMap shape_map;
  shape_map.emplace(
      "Weights",
      makeTensorInfo(
          ShapeInfo::DimType::CONSTANT, {1000, 58}, TensorProto_DataType_INT8));
  BoundShapeSpec spec(20, 1000);
  BoundShapeInferencer eng(spec);
  eng.InferBoundShapeAndType(net, shape_map);
  const auto& out_shape = eng.shape_info();
  verifyShapeInfo(
      out_shape,
      "Weights",
      ShapeInfo::DimType::CONSTANT,
      {1000, 58},
      TensorProto_DataType_INT8);
  verifyShapeInfo(
      out_shape,
      "Data",
      ShapeInfo::DimType::SEQ,
      {spec.max_seq_size},
      TensorProto_DataType_INT64);
  verifyShapeInfo(
      out_shape,
      "Lengths",
      ShapeInfo::DimType::BATCH,
      {spec.max_batch_size},
      TensorProto_DataType_INT32);
  verifyShapeInfo(
      out_shape, "Out", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 50});
}

TEST(BoundShapeInference, LengthsRangeFill) {
  NetDef net;
  net.add_op()->CopyFrom(CreateOperatorDef(
    "LengthsRangeFill",
    "",
    {"X"},
    {"Y"},
    {}));
  net.add_op()->CopyFrom(CreateOperatorDef(
    "Copy",
    "",
    {"Y"},
    {"Z"},
    {}));
  ShapeInfoMap shape_map;
  BoundShapeSpec spec(20, 1000);
  BoundShapeInferencer eng(spec);
  eng.InferBoundShapeAndType(net, shape_map);
  const auto& out_shape = eng.shape_info();
  verifyShapeInfo(
      out_shape,
      "X",
      ShapeInfo::DimType::BATCH,
      {spec.max_batch_size},
      TensorProto_DataType_INT32);
  verifyShapeInfo(
      out_shape,
      "Y",
      ShapeInfo::DimType::SEQ,
      {spec.max_seq_size},
      TensorProto_DataType_INT32);
  verifyShapeInfo(
      out_shape,
      "Z",
      ShapeInfo::DimType::SEQ,
      {spec.max_seq_size},
      TensorProto_DataType_INT32);
}

TEST(BoundShapeInference, Reshape) {
  NetDef net;
  std::vector<int> new_shape{-1, 8};
  std::vector<int> new_shape2{2, 8};
  net.add_op()->CopyFrom(
      CreateOperatorDef("FC", "", {"X0", "W0", "B0"}, {"X1"}, {}));
  net.add_op()->CopyFrom(CreateOperatorDef(
      "Reshape",
      "",
      {"X1"},
      {"Y1", "old_shape"},
      {MakeArgument<std::vector<int>>("shape", new_shape)}));

  // Cannot infer shape for this one because input/output shape doesn't match
  net.add_op()->CopyFrom(CreateOperatorDef(
      "Reshape",
      "",
      {"X1"},
      {"Y2", "old_shape2"},
      {MakeArgument<std::vector<int>>("shape", new_shape2)}));
  ShapeInfoMap shape_map;
  shape_map.emplace(
      "W0", makeTensorInfo(ShapeInfo::DimType::CONSTANT, {16, 1024}));
  shape_map.emplace("B0", makeTensorInfo(ShapeInfo::DimType::CONSTANT, {16}));
  BoundShapeSpec spec(20, 1000);
  BoundShapeInferencer eng(spec);
  eng.InferBoundShapeAndType(net, shape_map);
  const auto& out_shape = eng.shape_info();
  verifyShapeInfo(
      out_shape, "X0", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 1024});
  verifyShapeInfo(
      out_shape, "X1", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 16});
  verifyShapeInfo(
      out_shape,
      "Y1",
      ShapeInfo::DimType::BATCH,
      {spec.max_batch_size * 16 / 8, 8});
  EXPECT_TRUE(out_shape.find("Y2") == out_shape.end());
}

TEST(BoundShapeInference, ConcatMissingInput) {
  NetDef net;
  net.add_op()->CopyFrom(CreateOperatorDef(
      "Concat",
      "",
      {"I0", "I1"},
      {"Cout", "split_info"},
      {MakeArgument<int>("axis", 1), MakeArgument<int>("add_axis", 1)}));
  BoundShapeSpec spec(20, 1000);
  ShapeInfoMap shape_map;
  shape_map.emplace(
      "I0",
      makeTensorInfo(ShapeInfo::DimType::BATCH, {spec.max_batch_size, 60}));
  BoundShapeInferencer eng(spec);
  eng.InferBoundShapeAndType(net, shape_map);
  const auto& out_shape = eng.shape_info();
  verifyShapeInfo(
      out_shape, "I0", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 60});
  verifyShapeInfo(
      out_shape,
      "Cout",
      ShapeInfo::DimType::BATCH,
      {spec.max_batch_size, 2, 60});
}

TEST(BoundShapeInference, FC) {
  NetDef net;
  net.add_op()->CopyFrom(
      CreateOperatorDef("FC", "", {"X0", "W0", "B0"}, {"Out0"}, {}));
  net.add_op()->CopyFrom(
      CreateOperatorDef("FCTransposed", "", {"X1", "W1", "B1"}, {"Out1"}, {}));
  ShapeInfoMap shape_map;
  shape_map.emplace(
      "W0", makeTensorInfo(ShapeInfo::DimType::CONSTANT, {16, 1024}));
  shape_map.emplace("B0", makeTensorInfo(ShapeInfo::DimType::CONSTANT, {16}));
  shape_map.emplace(
      "W1", makeTensorInfo(ShapeInfo::DimType::CONSTANT, {16, 1024}));
  shape_map.emplace("B1", makeTensorInfo(ShapeInfo::DimType::CONSTANT, {1024}));
  BoundShapeSpec spec(20, 1000);
  BoundShapeInferencer eng(spec);
  eng.InferBoundShapeAndType(net, shape_map);
  const auto& out_shape = eng.shape_info();
  verifyShapeInfo(
      out_shape, "X0", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 1024});
  verifyShapeInfo(
      out_shape, "Out0", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 16});
  verifyShapeInfo(
      out_shape, "X1", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 16});
  verifyShapeInfo(
      out_shape,
      "Out1",
      ShapeInfo::DimType::BATCH,
      {spec.max_batch_size, 1024});
}

// We don't support inference input shape when Weight is not 2D
TEST(BoundShapeInference, UnsupportedFC) {
  NetDef net;
  net.add_op()->CopyFrom(
      CreateOperatorDef("FC", "", {"X0", "W0", "B0"}, {"Out0"}, {}));
  ShapeInfoMap shape_map;
  shape_map.emplace(
      "W0", makeTensorInfo(ShapeInfo::DimType::CONSTANT, {16, 1, 1024}));
  shape_map.emplace("B0", makeTensorInfo(ShapeInfo::DimType::CONSTANT, {16}));
  BoundShapeSpec spec(20, 1000);
  BoundShapeInferencer eng(spec);
  EXPECT_THROW(eng.InferBoundShapeAndType(net, shape_map), EnforceNotMet);
}

TEST(BoundShapeInference, Combo0) {
  NetDef net;
  net.add_op()->CopyFrom(CreateOperatorDef(
      "SparseLengthsSum", "", {"Weights0", "Data0", "Lengths0"}, {"EB0"}, {}));
  net.add_op()->CopyFrom(CreateOperatorDef(
      "SparseLengthsSum", "", {"Weights1", "Data1", "Lengths1"}, {"EB1"}, {}));
  net.add_op()->CopyFrom(CreateOperatorDef(
      "Concat",
      "",
      {"EB0", "EB1"},
      {"Cout", "split_info"},
      {MakeArgument<int>("axis", 1), MakeArgument<int>("add_axis", 1)}));
  net.add_op()->CopyFrom(CreateOperatorDef(
      "BatchMatMul",
      "",
      {"Cout", "Cout"},
      {"Bout"},
      {MakeArgument<int>("trans_b", 1)}));
  net.add_op()->CopyFrom(
      CreateOperatorDef("Flatten", "", {"Bout"}, {"Fout"}, {}));
  net.add_op()->CopyFrom(
      CreateOperatorDef("BatchGather", "", {"Fout", "Indices"}, {"Gout"}, {}));
  ShapeInfoMap shape_map;
  shape_map.emplace(
      "Weights0", makeTensorInfo(ShapeInfo::DimType::CONSTANT, {1000, 16}));
  shape_map.emplace(
      "Weights1", makeTensorInfo(ShapeInfo::DimType::CONSTANT, {20000, 16}));
  shape_map.emplace(
      "Indices", makeTensorInfo(ShapeInfo::DimType::CONSTANT, {2}));
  BoundShapeSpec spec(20, 1000);
  BoundShapeInferencer eng(spec);
  eng.InferBoundShapeAndType(net, shape_map);
  const auto& out_shape = eng.shape_info();
  LOG(INFO) << eng.PrintShapeInfo();
  verifyShapeInfo(
      out_shape, "Gout", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 2});
}
