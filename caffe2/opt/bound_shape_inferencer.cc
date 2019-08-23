#include "bound_shape_inferencer.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/tensor_impl.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

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
} // namespace

void BoundShapeInferencer::EnsureShapeNames(
    std::unordered_map<std::string, ShapeInfo>* info) const {
  for (auto& kv : *info) {
    kv.second.shape.set_name(kv.first);
  }
}

void BoundShapeInferencer::InferOps(
    const OperatorDef& op,
    caffe2::Workspace* /* ws */) {
  if (op.type() == "SparseLengthsSum" ||
      op.type() == "SparseLengthsSumFused8BitRowwise" ||
      op.type() == "SparseLengthsWeightedSum" ||
      op.type() == "SparseLengthsWeightedSumFused8BitRowwise") {
    InferSparseLengthsSum(op);
  } else if (
      op.type() == "FC" || op.type() == "FCTransposed" ||
      op.type() == "FbFCPacked") {
    InferFC(op);
  } else if (op.type() == "Concat") {
    InferConcat(op);
  } else if (op.type() == "Reshape") {
    InferReshape(op);
  } else if (op.type() == "LengthsRangeFill") {
    InferLengthsRangeFill(op);
  } else if (
      (caffe2::StartsWith(op.type(), "GivenTensor") &&
       caffe2::EndsWith(op.type(), "Fill")) ||
      op.type() == "ConstantFill" || op.type() == "Int8GivenTensorFill" ||
      op.type() == "Int8GivenIntTensorFill") {
    InferGivenTensorFill(op);
  } else if (op.type() == "Shape") {
    InferShape(op);
  } else {
    InferCommonOp(op);
  }
}

void BoundShapeInferencer::InferBoundShapeAndType(
    const NetDef& net,
    const std::unordered_map<std::string, ShapeInfo>& info,
    caffe2::Workspace* ws) {
  const static std::unordered_set<std::string> unsupported{"Tile"};
  shape_info_ = info;

  bool inferFinished = false;

  auto old_shape_num = shape_info_.size();
  while (!inferFinished) {
    for (const auto& op : net.op()) {
      VLOG(1) << op.type();
      if (unsupported.count(op.type())) {
        continue;
      }
      InferOps(op, ws);
    }

    // Doing a reverse pass to infer the input shapes if applicable
    for (int i = net.op_size() - 1; i >= 0; --i) {
      const auto& op = net.op(i);
      if (op.type() == "Concat") {
        InferConcatInputs(op);
      }
    }
    inferFinished = old_shape_num == shape_info_.size();
    LOG(INFO) << "old shape info num: " << old_shape_num
              << ", new shape info num: " << shape_info_.size();
    old_shape_num = shape_info_.size();
  }

  // Make sure shape has name
  EnsureShapeNames(&shape_info_);
}

TensorShape& BoundShapeInferencer::SetTensorShapeAndTypeIfNotExist(
    const std::string& name,
    ShapeInfo::DimType t,
    std::vector<int64_t> bound_dims,
    TensorProto::DataType type,
    bool is_quantized) {
  return CheckAndSetTensorShapeAndType(
      name, t, bound_dims, type, is_quantized, true);
}

// if allow_existing_shape is true, we use existing shape directly
// and not enforce shape to be equal to bound_dims
// else we enforce them to be equal
TensorShape& BoundShapeInferencer::CheckAndSetTensorShapeAndType(
    const std::string& name,
    ShapeInfo::DimType t,
    std::vector<int64_t> bound_dims,
    TensorProto::DataType type,
    bool is_quantized,
    bool allow_existing_shape) {
  auto rt = shape_info_.emplace(name, ShapeInfo());
  ShapeInfo& shape_info = rt.first->second;
  TensorShape& shape = shape_info.shape;
  if (is_quantized) {
    shape_info.is_quantized = true;
    shape_info.q_info.scale.clear();
    shape_info.q_info.scale.push_back(1);
    shape_info.q_info.offset.clear();
    shape_info.q_info.offset.push_back(0);
    shape_info.q_info.axis = 1;
  }
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

    if (!allow_existing_shape) {
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

void BoundShapeInferencer::InferGivenTensorFill(const OperatorDef& op) {
  CAFFE_ENFORCE_EQ(op.output_size(), 1, op.type(), " must have 1 output");
  InferCommonOp(op);
  auto it = shape_info_.find(op.output(0));
  if (it != shape_info_.end()) {
    it->second.dim_type = ShapeInfo::DimType::CONSTANT;
  }
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
      TensorProto_DataType_INT32,
      false);
  CheckAndSetTensorShapeAndType(
      op.output(0),
      ShapeInfo::DimType::SEQ,
      {spec_.max_seq_size},
      TensorProto_DataType_INT32,
      false);
  current_dim_type_ = ShapeInfo::DimType::SEQ;
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
    SetTensorShapeAndTypeIfNotExist(
        op.input(weight),
        ShapeInfo::DimType::SEQ,
        {spec_.max_seq_size},
        TensorProto_DataType_FLOAT,
        false);
  }

  // Bound inputs
  SetTensorShapeAndTypeIfNotExist(
      op.input(1 + weight),
      ShapeInfo::DimType::SEQ,
      {spec_.max_seq_size},
      TensorProto_DataType_INT64,
      false);
  CheckAndSetTensorShapeAndType(
      op.input(2 + weight),
      ShapeInfo::DimType::BATCH,
      {spec_.max_batch_size},
      TensorProto_DataType_INT32,
      false);

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
      TensorProto_DataType_FLOAT,
      false);
}

void BoundShapeInferencer::InferShape(const OperatorDef& op) {
  InferCommonOp(op);
  // old_shape should be a constant
  if (op.output_size() > 0 && shape_info_.count(op.output(0))) {
    shape_info_[op.output(0)].dim_type = ShapeInfo::DimType::CONSTANT;
  }
}

void BoundShapeInferencer::InferReshape(const OperatorDef& op) {
  InferCommonOp(op);
  // old_shape should be a constant
  if (op.output_size() > 1 && shape_info_.count(op.output(1))) {
    shape_info_[op.output(1)].dim_type = ShapeInfo::DimType::CONSTANT;
  }
}

void BoundShapeInferencer::InferConcatInputs(const OperatorDef& op) {
  ArgumentHelper helper(op);
  const auto add_axis = helper.GetSingleArgument<int32_t>("add_axis", 0);
  if (add_axis) {
    return;
  } else if (op.output_size() == 0 || !shape_info_.count(op.output(0))) {
    return;
  }

  const auto axis = helper.HasArgument("axis")
      ? helper.GetSingleArgument<int32_t>("axis", -1)
      : GetDimFromOrderString(
            helper.GetSingleArgument<string>("order", "NCHW"));

  const auto& shape_info = shape_info_.at(op.output(0));
  int output_channel = shape_info.shape.dims(axis);
  int missing_shape_infos = 0;
  int channel_acc = 0;
  std::string input_to_infer;
  for (const auto& i : op.input()) {
    const auto it = shape_info_.find(i);
    if (it != shape_info_.end()) {
      const auto& current_input_shape = it->second;
      if (axis < current_input_shape.shape.dims_size()) {
        channel_acc += current_input_shape.shape.dims(axis);
      } else {
        LOG(INFO) << "Mismatched input dim along axis " << axis
                  << ". We cannot infer missing input shape for Concat";
        return;
      }
    } else if (missing_shape_infos) {
      LOG(INFO) << "More than one missing shapes, previous one: "
                << input_to_infer;
      // We can only infer one missing input shape info
      return;
    } else {
      ++missing_shape_infos;
      input_to_infer = i;
    }
  }

  if (missing_shape_infos && !input_to_infer.empty()) {
    auto input_shape_info = shape_info;
    input_shape_info.shape.set_dims(axis, output_channel - channel_acc);
    shape_info_.emplace(input_to_infer, std::move(input_shape_info));

    // Infer the shape of the second output of Concat
    InferCommonOp(op);
    if (op.output_size() > 1 && shape_info_.count(op.output(1))) {
      shape_info_[op.output(1)].dim_type = ShapeInfo::DimType::CONSTANT;
    }
  }
}

// For concat net, if some inputs are missing and we have add_axis argument,
// it means that all the inputs should be of the same dimension. In this case,
// we can infer the shape of the missing inputs
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
  bool fp16 = (op.type() == "FbFCPacked");
  auto x_it = shape_info_.find(op.input(0));
  if (x_it == shape_info_.end()) {
    // We don't have a hint at the x input we try to deduce it from weight
    // shape
    ArgumentHelper helper(op);
    auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
    auto axis_w = helper.GetSingleArgument<int32_t>("axis_w", 1);
    const TensorShape w_shape = w_shape_info.shape;
    bool transposed = (op.type() == "FCTransposed") ? true : false;
    const int canonical_axis_w =
        canonical_axis_index_(axis_w, w_shape.dims().size());
    const int64_t K = transposed ? SizeToDim(w_shape, canonical_axis_w)
                                 : SizeFromDim(w_shape, canonical_axis_w);
    std::vector<int64_t> dims;
    for (int i = 0; i < axis - 1; ++i) {
      dims.push_back(1);
    }
    dims.push_back(spec_.max_batch_size);
    dims.push_back(K);
    current_dim_type_ = ShapeInfo::DimType::BATCH;
    current_max_batch_size_ = spec_.max_batch_size;
    // Note: for FbFCPacked, weight is fp16 but actications are in fp32
    CheckAndSetTensorShapeAndType(
        op.input(0),
        ShapeInfo::DimType::BATCH,
        dims,
        fp16 ? TensorProto_DataType_FLOAT : w_shape.data_type(),
        false);
  } else {
    ShapeInfo& x_shape_info = x_it->second;
    if (x_shape_info.dim_type != ShapeInfo::DimType::BATCH) {
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
      fp16 ? TensorProto_DataType_FLOAT : output_shapes[0].data_type(),
      false);
}

void BoundShapeInferencer::InferCommonOp(const OperatorDef& op) {
  // First, we need to check that all the input shape/types are already
  // presented
  try {
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
    output_shapes = schema->InferTensor(op, input_shapes);
    int i = 0;
    bool is_quantized =
        !(op.type().compare(0, 4, "Int8")) && (op.type() != "Int8Dequantize");
    TensorProto::DataType infered_data_type = TensorProto::UNDEFINED;
    if (is_quantized) {
      const static std::map<std::string, int> type_info_from_input = {
          {"Int8Quantize", -1}, // Force this op's output to be uint8
          {"Int8ConvRelu", 1},
          {"Int8MaxPool", 0},
          {"Int8AveragePool", 0},
          {"Int8FC", 1},
          {"Int8Conv", 1},
          {"Int8SumRelu", 0}};
      CAFFE_ENFORCE(
          type_info_from_input.find(op.type()) != type_info_from_input.end(),
          "Undefined quantized output data type, add it into type_info_from_input");
      int target = type_info_from_input.find(op.type())->second;
      if (target == -1) {
        infered_data_type = TensorProto::UINT8;
      } else {
        CAFFE_ENFORCE(target < input_shapes.size());
        infered_data_type = input_shapes[target].data_type();
      }
    } else if (op.type() == "Int8Dequantize") {
      infered_data_type = TensorProto::FLOAT;
    }

    for (const auto& shape : output_shapes) {
      if (infered_data_type == TensorProto::UNDEFINED) {
        infered_data_type = shape.data_type();
      }
      if (shape.unknown_shape()) {
        ++i;
        continue;
      }
      CheckAndSetTensorShapeAndType(
          op.output(i++),
          current_dim_type_,
          ConvertToVec(shape.dims()),
          infered_data_type,
          is_quantized);
    }
  } catch (const caffe2::EnforceNotMet& e) {
    LOG(ERROR) << "Enforce not met while inferring shapes for " << op.type()
               << ": " << e.msg();
  } catch (const std::exception& e) {
    LOG(WARNING) << "Caught exception while inferring shapes for " << op.type()
                 << ": " << e.what();
  }
}

std::shared_ptr<BoundShapeInferencerBase> getBoundShapeInferencer(
    const BoundShapeSpec& spec) {
  return std::make_shared<BoundShapeInferencer>(spec);
}

C10_DEFINE_SHARED_REGISTRY(
    BoundShapeInferencerRegistry,
    BoundShapeInferencerBase,
    const BoundShapeSpec&);

C10_REGISTER_CREATOR(
    BoundShapeInferencerRegistry,
    C10,
    getBoundShapeInferencer);
} // namespace caffe2
