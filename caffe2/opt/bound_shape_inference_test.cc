#include <gtest/gtest.h>
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/opt/bound_shape_inferencer.h"
#include "caffe2/utils/proto_utils.h"

using namespace caffe2;
namespace {
using ShapeInfoMap = std::unordered_map<std::string, ShapeInfo>;

ShapeInfo MakeTensorInfo(
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

void PrintShape(const ShapeInfoMap& map) {
  for (const auto& kv : map) {
    const auto& s = kv.second;
    std::stringstream ss;
    ss << s.shape.name() << ": dim_type: " << s.dim_type << ", dims: [";
    for (const auto d : s.shape.dims()) {
      ss << d << ", ";
    }
    ss << "], dtype: " << s.shape.data_type();
    LOG(INFO) << ss.str();
  }
}

void VerifyShapeInfo(
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
      "Weights", MakeTensorInfo(ShapeInfo::DimType::CONSTANT, {16, 1000}));
  BoundShapeSpec spec(20, 1000);
  BoundShapeInferencer eng(spec);
  eng.InferBoundShapeAndType(net, shape_map);
  const auto& out_shape = eng.shape_info();
  VerifyShapeInfo(
      out_shape, "Weights", ShapeInfo::DimType::CONSTANT, {16, 1000});
  VerifyShapeInfo(
      out_shape,
      "Data",
      ShapeInfo::DimType::SEQ,
      {spec.max_seq_size},
      TensorProto_DataType_INT32);
  VerifyShapeInfo(
      out_shape,
      "Lengths",
      ShapeInfo::DimType::BATCH,
      {spec.max_batch_size},
      TensorProto_DataType_INT32);
  VerifyShapeInfo(
      out_shape, "Out", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 16});
}

TEST(BoundShapeInference, FC) {
  NetDef net;
  net.add_op()->CopyFrom(
      CreateOperatorDef("FC", "", {"X0", "W0", "B0"}, {"Out0"}, {}));
  net.add_op()->CopyFrom(
      CreateOperatorDef("FCTransposed", "", {"X1", "W1", "B1"}, {"Out1"}, {}));
  ShapeInfoMap shape_map;
  shape_map.emplace(
      "W0", MakeTensorInfo(ShapeInfo::DimType::CONSTANT, {16, 1024}));
  shape_map.emplace("B0", MakeTensorInfo(ShapeInfo::DimType::CONSTANT, {16}));
  shape_map.emplace(
      "W1", MakeTensorInfo(ShapeInfo::DimType::CONSTANT, {16, 1024}));
  shape_map.emplace("B1", MakeTensorInfo(ShapeInfo::DimType::CONSTANT, {1024}));
  BoundShapeSpec spec(20, 1000);
  BoundShapeInferencer eng(spec);
  eng.InferBoundShapeAndType(net, shape_map);
  const auto& out_shape = eng.shape_info();
  VerifyShapeInfo(
      out_shape, "X0", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 1024});
  VerifyShapeInfo(
      out_shape, "Out0", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 16});
  VerifyShapeInfo(
      out_shape, "X1", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 16});
  VerifyShapeInfo(
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
      "W0", MakeTensorInfo(ShapeInfo::DimType::CONSTANT, {16, 1, 1024}));
  shape_map.emplace("B0", MakeTensorInfo(ShapeInfo::DimType::CONSTANT, {16}));
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
      "Weights0", MakeTensorInfo(ShapeInfo::DimType::CONSTANT, {16, 1000}));
  shape_map.emplace(
      "Weights1", MakeTensorInfo(ShapeInfo::DimType::CONSTANT, {16, 20000}));
  shape_map.emplace(
      "Indices", MakeTensorInfo(ShapeInfo::DimType::CONSTANT, {2}));
  BoundShapeSpec spec(20, 1000);
  BoundShapeInferencer eng(spec);
  eng.InferBoundShapeAndType(net, shape_map);
  const auto& out_shape = eng.shape_info();
  PrintShape(out_shape);
  VerifyShapeInfo(
      out_shape, "Gout", ShapeInfo::DimType::BATCH, {spec.max_batch_size, 2});
}
