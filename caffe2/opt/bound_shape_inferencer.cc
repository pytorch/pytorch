#include "bound_shape_inferencer.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/tensor_impl.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

namespace {
std::vector<int64_t> ConvertToVec(
    const ::google::protobuf::RepeatedField<::google::protobuf::int64>& in) {
  std::vector<int64_t> out;
  out.reserve(in.size());
  for (const auto d : in) {
    out.push_back(d);
  }
  return out;
}

int64_t SizeFromDim(const TensorShape& shape, int axis) {
  int64_t r = 1;
  for (int i = axis; i < shape.dims_size(); ++i) {
    r *= shape.dims(i);
  }
  return r;
}

int64_t SizeToDim(const TensorShape& shape, int axis) {
  CAFFE_ENFORCE_LE(axis, shape.dims_size());
  int64_t r = 1;
  for (int i = 0; i < axis; ++i) {
    r *= shape.dims(i);
  }
  return r;
}

void EnsureShapeNames(std::unordered_map<std::string, ShapeInfo>* info) {
  for (auto& kv : *info) {
    kv.second.shape.set_name(kv.first);
  }
}
} // namespace

void BoundShapeInferencer::InferBoundShapeAndType(
    const NetDef& net,
    const std::unordered_map<std::string, ShapeInfo>& info) {
  shape_info_ = info;
  visited_tensors_.clear();

  for (const auto& op : net.op()) {
    if (op.type() == "SparseLengthsSum" ||
        op.type() == "SparseLengthsSumFused8BitRowwise") {
      InferSparseLengthsSum(op);
    } else if (op.type() == "FC" || op.type() == "FCTransposed") {
      InferFC(op);
    } else {
      InferCommonOp(op);
    }
  }

  // Make sure shape has name
  EnsureShapeNames(&shape_info_);
}

TensorShape& BoundShapeInferencer::CheckAndSetTensorShapeAndType(
    const std::string& name,
    ShapeInfo::DimType t,
    std::vector<int64_t> bound_dims,
    TensorProto::DataType type) {
  if (!visited_tensors_.emplace(name).second) {
    return shape_info_.at(name).shape;
  }
  auto rt = shape_info_.emplace(name, ShapeInfo());
  ShapeInfo& shape_info = rt.first->second;
  shape_info.dim_type = t;
  TensorShape& shape = shape_info.shape;
  if (!rt.second) {
    // Check shape consistency
    CAFFE_ENFORCE_EQ(shape.dims_size(), bound_dims.size());
    // For shapes that was provided as a hint at the input of the net, fix the
    // batch size first.
    if (shape.dims_size() > 0 &&
        shape_info.dim_type == ShapeInfo::DimType::UNKNOWN &&
        t > ShapeInfo::DimType::CONSTANT) {
      shape_info.dim_type = t;
      shape.set_dims(0, bound_dims.front());
    }
    for (int i = 0; i < shape.dims_size(); ++i) {
      CAFFE_ENFORCE_EQ(
          shape.dims(i),
          bound_dims[i],
          "Shape inconsistency found in tensor ",
          name,
          " on dim ",
          i,
          " (",
          shape.dims(i),
          " vs ",
          bound_dims[i],
          ")");
    }
    return shape;
  }

  shape.mutable_dims()->Clear();
  for (const auto d : bound_dims) {
    shape.add_dims(d);
  }
  shape.set_data_type(type);
  return shape;
}

std::vector<TensorShape> InferOutput(
    const OperatorDef& op,
    const std::vector<TensorShape>& input_shapes) {
  const OpSchema* schema = OpSchemaRegistry::Schema(op.type());
  CAFFE_ENFORCE(schema);
  return schema->InferTensor(op, input_shapes);
}

void BoundShapeInferencer::InferSparseLengthsSum(const OperatorDef& op) {
  CAFFE_ENFORCE_EQ(op.input_size(), 3, "SparseLengthsSum has to have 3 inputs");
  const auto it = shape_info_.find(op.input(0));
  CAFFE_ENFORCE(
      it != shape_info_.end(),
      "Shape of DATA input of SparseLengthsSum ",
      op.input(0),
      " needs to be presented");

  // Bound inputs
  CheckAndSetTensorShapeAndType(
      op.input(1),
      ShapeInfo::DimType::SEQ,
      {spec_.max_seq_size},
      TensorProto_DataType_INT32);
  CheckAndSetTensorShapeAndType(
      op.input(2),
      ShapeInfo::DimType::BATCH,
      {spec_.max_batch_size},
      TensorProto_DataType_INT32);

  // Infer output
  CAFFE_ENFORCE_EQ(it->second.shape.dims_size(), 2);
  current_dim_type_ = ShapeInfo::DimType::BATCH;
  current_max_batch_size_ = spec_.max_batch_size;
  CheckAndSetTensorShapeAndType(
      op.output(0),
      ShapeInfo::DimType::BATCH,
      {spec_.max_batch_size, it->second.shape.dims(0)},
      it->second.shape.data_type());
}

void BoundShapeInferencer::InferFC(const OperatorDef& op) {
  CAFFE_ENFORCE_EQ(op.input_size(), 3, "FC has to have 3 inputs");
  const auto w_it = shape_info_.find(op.input(1));
  CAFFE_ENFORCE(
      w_it != shape_info_.end(),
      "Shape of WEIGHT input of FC ",
      op.input(1),
      " needs to be presented");
  const ShapeInfo& w_shape_info = w_it->second;
  const auto b_it = shape_info_.find(op.input(2));
  CAFFE_ENFORCE(
      w_it != shape_info_.end(),
      "Shape of BIAS input of FC ",
      op.input(2),
      " needs to be presented");
  const ShapeInfo& b_shape_info = b_it->second;
  auto x_it = shape_info_.find(op.input(0));
  if (x_it == shape_info_.end()) {
    // We don't have a hint at the x input we try to deduce it from weight shape
    ArgumentHelper helper(op);
    auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
    auto axis_w = helper.GetSingleArgument<int32_t>("axis_w", 1);
    CAFFE_ENFORCE_EQ(
        axis,
        1,
        "Don't know how to deduce input of FC with axis not equal to 1: ",
        op.input(0));
    CAFFE_ENFORCE_EQ(
        axis_w,
        1,
        "Don't know how to deduce input of FC with axis_w not equal to 1: ",
        op.input(0));
    const TensorShape w_shape = w_shape_info.shape;
    CAFFE_ENFORCE_EQ(
        w_shape.dims_size(),
        2,
        "Don't know how to deduce input of FC other than of dim size 2: ",
        op.input(0));
    bool transposed = (op.type() == "FC") ? false : true;
    const int canonical_axis_w =
        canonical_axis_index_(axis_w, w_shape.dims().size());
    const int64_t K = transposed ? SizeToDim(w_shape, canonical_axis_w)
                                 : SizeFromDim(w_shape, canonical_axis_w);
    current_dim_type_ = ShapeInfo::DimType::BATCH;
    current_max_batch_size_ = spec_.max_batch_size;
    CheckAndSetTensorShapeAndType(
        op.input(0),
        ShapeInfo::DimType::BATCH,
        {spec_.max_batch_size, K},
        w_shape.data_type());
  } else {
    ShapeInfo& x_shape_info = x_it->second;
    if (x_shape_info.dim_type == ShapeInfo::DimType::UNKNOWN) {
      CAFFE_ENFORCE_GE(x_shape_info.shape.dims_size(), 1);
      x_shape_info.shape.set_dims(0, spec_.max_batch_size);
      x_shape_info.dim_type = ShapeInfo::DimType::BATCH;
    }
  }

  // Standard shape inference for outputs
  std::vector<TensorShape> input_shapes{
      shape_info_[op.input(0)].shape, w_shape_info.shape, b_shape_info.shape};
  std::vector<TensorShape> output_shapes = InferOutput(op, input_shapes);
  CAFFE_ENFORCE_EQ(output_shapes.size(), 1);
  CheckAndSetTensorShapeAndType(
      op.output(0),
      ShapeInfo::DimType::BATCH,
      ConvertToVec(output_shapes[0].dims()),
      output_shapes[0].data_type());
}

void BoundShapeInferencer::InferCommonOp(const OperatorDef& op) {
  // First, we need to check that all the input shape/types are already
  // presented
  std::vector<TensorShape> input_shapes;
  for (const auto& input : op.input()) {
    const auto it = shape_info_.find(input);
    CAFFE_ENFORCE(it != shape_info_.end());
    input_shapes.emplace_back(it->second.shape);
  }

  const OpSchema* schema = OpSchemaRegistry::Schema(op.type());
  CAFFE_ENFORCE(schema);
  auto output_shapes = schema->InferTensor(op, input_shapes);
  CAFFE_ENFORCE_EQ(output_shapes.size(), op.output_size());
  int i = 0;
  for (const auto& shape : output_shapes) {
    CheckAndSetTensorShapeAndType(
        op.output(i++),
        current_dim_type_,
        ConvertToVec(shape.dims()),
        shape.data_type());
  }
}

} // namespace caffe2
