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

  for (const auto& op : net.op()) {
    LOG(INFO) << op.type();
    if (op.type() == "SparseLengthsSum" ||
        op.type() == "SparseLengthsSumFused8BitRowwise" ||
        op.type() == "SparseLengthsWeightedSum" ||
        op.type() == "SparseLengthsWeightedSumFused8BitRowwise") {
      InferSparseLengthsSum(op);
    } else if (op.type() == "FC" || op.type() == "FCTransposed") {
      InferFC(op);
    } else if (op.type() == "Concat") {
      InferConcat(op);
    } else if (op.type() == "Reshape") {
      InferReshape(op);
    } else if (op.type() == "LengthsRangeFill") {
      InferLengthsRangeFill(op);
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
  auto rt = shape_info_.emplace(name, ShapeInfo());
  ShapeInfo& shape_info = rt.first->second;
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

  shape_info.dim_type = t;
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

void BoundShapeInferencer::InferLengthsRangeFill(const OperatorDef& op) {
  CAFFE_ENFORCE_EQ(op.input_size(), 1, "LengthsRangeFill must have 1 input");
  CAFFE_ENFORCE_EQ(op.output_size(), 1, "LengthsRangeFill must have 1 output");
  // Both input and ouptut of LengthsRangeFill is int32:
  // https://fburl.com/fhwb5666
  CheckAndSetTensorShapeAndType(
      op.input(0),
      ShapeInfo::DimType::BATCH,
      {spec_.max_batch_size},
      TensorProto_DataType_INT32);
  CheckAndSetTensorShapeAndType(
      op.output(0),
      ShapeInfo::DimType::SEQ,
      {spec_.max_seq_size},
      TensorProto_DataType_INT32);
}

void BoundShapeInferencer::InferSparseLengthsSum(const OperatorDef& op) {
  CAFFE_ENFORCE_GE(
      op.input_size(), 3, op.type(), " must have at least 3 inputs");
  const auto it = shape_info_.find(op.input(0));
  CAFFE_ENFORCE(
      it != shape_info_.end(),
      "Shape of DATA input of SparseLengthsSum ",
      op.input(0),
      " needs to be presented");
  CAFFE_ENFORCE_EQ(
      it->second.shape.dims().size(),
      2,
      "DATA input ",
      op.input(0),
      "needs to be 2D");

  int weight = (op.type() == "SparseLengthsWeightedSum" ||
                op.type() == "SparseLengthsWeightedSumFused8BitRowwise")
      ? 1
      : 0;

  if (weight) {
    CAFFE_ENFORCE_EQ(
        op.input_size(), 4, "SparseLengthsWeightedSum must have 4 inputs");
    CheckAndSetTensorShapeAndType(
        op.input(weight),
        ShapeInfo::DimType::SEQ,
        {spec_.max_seq_size},
        TensorProto_DataType_FLOAT);
  }

  // Bound inputs
  CheckAndSetTensorShapeAndType(
      op.input(1 + weight),
      ShapeInfo::DimType::SEQ,
      {spec_.max_seq_size},
      TensorProto_DataType_INT64);
  CheckAndSetTensorShapeAndType(
      op.input(2 + weight),
      ShapeInfo::DimType::BATCH,
      {spec_.max_batch_size},
      TensorProto_DataType_INT32);

  // Infer output
  CAFFE_ENFORCE_EQ(it->second.shape.dims_size(), 2);
  current_dim_type_ = ShapeInfo::DimType::BATCH;
  current_max_batch_size_ = spec_.max_batch_size;
  auto output_dim1 = it->second.shape.dims(1);
  // If the op is SparseLengthsSumFused8BitRowwise, we need to extract 4 for
  // scale and 4 byte for bias (https://fburl.com/t6dp9tsc)
  if (op.type() == "SparseLengthsSumFused8BitRowwise" ||
      op.type() == "SparseLengthsWeightedSumFused8BitRowwise") {
    output_dim1 -= 8;
  }
  CheckAndSetTensorShapeAndType(
      op.output(0),
      ShapeInfo::DimType::BATCH,
      {spec_.max_batch_size, output_dim1},
      TensorProto_DataType_FLOAT);
}

void BoundShapeInferencer::InferReshape(const OperatorDef& op) {
  InferCommonOp(op);
  // old_shape should be a constant
  if (op.output_size() > 1 && shape_info_.count(op.output(1))) {
    shape_info_[op.output(1)].dim_type = ShapeInfo::DimType::CONSTANT;
  }
}
// For concat net, if some inputs are missing and we have add_axis argument, it
// means that all the inputs should be of the same dimension. In this case, we
// can infer the shape of the missing inputs
void BoundShapeInferencer::InferConcat(const OperatorDef& op) {
  ArgumentHelper helper(op);
  auto add_axis = helper.GetSingleArgument<int32_t>("add_axis", 0);
  if (add_axis) {
    ShapeInfo* ref_input_shape = nullptr;
    std::string ref_name;
    std::unordered_set<std::string> missing_shape_inputs;
    for (const auto& i : op.input()) {
      const auto it = shape_info_.find(i);
      if (it != shape_info_.end()) {
        const auto& current_input_shape = it->second;
        if (ref_input_shape) {
          CAFFE_ENFORCE_EQ(
              ref_input_shape->shape.dims_size(),
              current_input_shape.shape.dims_size(),
              ref_name,
              " vs ",
              i);
          for (int j = 0; j < ref_input_shape->shape.dims_size(); ++j) {
            CAFFE_ENFORCE_EQ(
                ref_input_shape->shape.dims(j),
                current_input_shape.shape.dims(j),
                "Mismatched size on dim ",
                j,
                " between ",
                ref_name,
                " and ",
                i,
                " (",
                ref_input_shape->shape.dims(j),
                " vs ",
                current_input_shape.shape.dims(j),
                ")");
          }
        } else {
          ref_input_shape = &it->second;
          ref_name = i;
        }
      } else {
        missing_shape_inputs.emplace(i);
      }
    }

    if (ref_input_shape) {
      current_dim_type_ = ref_input_shape->dim_type;
      for (const auto& i : missing_shape_inputs) {
        shape_info_.emplace(i, *ref_input_shape);
      }
    }
  }
  InferCommonOp(op);
  // split_info should be a constant
  if (op.output_size() > 1 && shape_info_.count(op.output(1))) {
    shape_info_[op.output(1)].dim_type = ShapeInfo::DimType::CONSTANT;
  }
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
    if (it == shape_info_.end()) {
      LOG(WARNING) << "Cannot find shape info for " << input << ". Skipping "
                   << op.type();
      return;
    }
    input_shapes.emplace_back(it->second.shape);
  }

  const OpSchema* schema = OpSchemaRegistry::Schema(op.type());
  CAFFE_ENFORCE(schema);
  std::vector<TensorShape> output_shapes;
  try {
    output_shapes = schema->InferTensor(op, input_shapes);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Caught exception while inferring shapes for " << op.type();
  }
  int i = 0;
  for (const auto& shape : output_shapes) {
    if (shape.unknown_shape()) {
      ++i;
      continue;
    }
    CheckAndSetTensorShapeAndType(
        op.output(i++),
        current_dim_type_,
        ConvertToVec(shape.dims()),
        shape.data_type());
  }
}

} // namespace caffe2
