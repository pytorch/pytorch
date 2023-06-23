#include "bound_shape_inferencer.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/tensor_impl.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

#include <c10/util/irange.h>

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

std::vector<TensorBoundShape::DimType> setDimTypeWithFirst(
    TensorBoundShape::DimType firstDimType,
    uint32_t n) {
  std::vector<TensorBoundShape::DimType> dimTypes(
      n, TensorBoundShape_DimType_CONSTANT);
  if (dimTypes.size() > 0) {
    dimTypes[0] = firstDimType;
  }
  return dimTypes;
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

// Check precedence between two vector of ensorBoundShape::DimType.
// If return 1: right take precedence over left
// If return -1: left take precedence over right
// If return 0: no precedence between left and right
int takePrecedenceOver(
    const std::vector<TensorBoundShape::DimType>& left,
    const std::vector<TensorBoundShape::DimType>& right) {
  const static std::vector<
      std::tuple<TensorBoundShape::DimType, TensorBoundShape::DimType, int>>
      precedence = {
          std::tuple<TensorBoundShape::DimType, TensorBoundShape::DimType, int>{
              TensorBoundShape_DimType_FEATURE_MAX_DEFAULT,
              TensorBoundShape_DimType_FEATURE_MAX,
              1},
          std::tuple<TensorBoundShape::DimType, TensorBoundShape::DimType, int>{
              TensorBoundShape_DimType_FEATURE_MAX,
              TensorBoundShape_DimType_FEATURE_MAX_DEFAULT,
              -1},
          std::tuple<TensorBoundShape::DimType, TensorBoundShape::DimType, int>{
              TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT,
              TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX,
              1},
          std::tuple<TensorBoundShape::DimType, TensorBoundShape::DimType, int>{
              TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX,
              TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT,
              -1}};

  // If left is empty and right is not, right take precedence
  if (left.size() == 0 || right.size() == 0) {
    return right.size() > left.size();
  }
  for (auto i: c10::irange(right.size())) {
    // If right.size > left.size and left[0:i] == right[0:i],
    // right take precedence
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    if (i >= left.size()) {
      return 1;
    }
    auto l = left[i];
    auto r = right[i];
    if (l == TensorBoundShape_DimType_UNKNOWN &&
        r != TensorBoundShape_DimType_UNKNOWN) {
      return 1;
    }
    if (r == TensorBoundShape_DimType_UNKNOWN &&
        l != TensorBoundShape_DimType_UNKNOWN) {
      return -1;
    }
    for (auto& t : precedence) {
      if (l == std::get<0>(t) && r == std::get<1>(t)) {
        return std::get<2>(t);
      }
    }
    if (l != r) {
      return 0;
    }
  }
  return 0;
}
} // namespace

void BoundShapeInferencer::EnsureShapeNames(
    std::unordered_map<std::string, ShapeInfo>* info) const {
  for (auto& kv : *info) {
    kv.second.shape.set_name(kv.first);
  }
}

void BoundShapeInferencer::Initialize(
    const ShapeInfoMap& info,
    bool extract_feature_len) {
  shape_info_ = info;
  extract_feature_len_ = extract_feature_len;
}

void BoundShapeInferencer::InferOps(
    const OperatorDef& op,
    caffe2::Workspace* /* ws */) {
  const static std::unordered_set<std::string> kSlsOps = {
      "SparseLengthsSum",
      "SparseLengthsSumFused8BitRowwise",
      "SparseLengthsWeightedSum",
      "SparseLengthsWeightedSumFused8BitRowwise",
      "SparseLengthsSumFused4BitRowwise",
      "SparseLengthsWeightedSumFused4BitRowwise",
      "SparseLengthsSum4BitRowwiseSparse",
      "SparseLengthsWeightedSum4BitRowwiseSparse",
      "SparseLengthsSum8BitRowwiseSparse",
      "SparseLengthsWeightedSum8BitRowwiseSparse"};
  if (kSlsOps.count(op.type())) {
    InferSparseLengthsSum(op);
  } else if (op.type() == "Add" || op.type() == "Mul") {
    InferElementwiseOp(op);
  } else if (
      op.type() == "FC" || op.type() == "FCTransposed" ||
      op.type() == "FbFCPacked" || op.type() == "Int8FC") {
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
  } else if (
      op.type() == "FloatToFused8BitRowwiseQuantized" ||
      op.type() == "HalfFloatToFused8BitRowwiseQuantized" ||
      op.type() == "FloatToFused4BitRowwiseQuantized" ||
      op.type() == "HalfToFused4BitRowwiseQuantized" ||
      op.type() == "FloatToHalf" || op.type() == "FbGemmPack") {
    InferQuantizationTransformation(op);
  } else if (op.type() == "UnPackRecords") {
    InferUnPackRecords(op);
  } else if (op.type() == "Tile") {
    InferTile(op);
  } else if (op.type() == "SparseLengthsSumSparseLookup") {
    InferSparseLengthsSumSparseLookup(op);
  } else if (op.type() == "Softmax") {
    InferSoftmax(op);
  } else if (op.type() == "LpNorm") {
    InferLpNorm(op);
  } else if (op.type() == "Transpose") {
    InferTranspose(op);
  } else if (op.type() == "Bucketize") {
    InferBucketize(op);
  } else if (op.type() == "Clip") {
    InferClip(op);
  } else if (op.type() == "Div") {
    InferDiv(op);
  } else if (op.type() == "Mean") {
    InferMean(op);
  } else {
    InferCommonOp(op);
  }
}

void BoundShapeInferencer::InferBoundShapeAndType(
    const NetDef& net,
    const ShapeInfoMap& info,
    caffe2::Workspace* ws,
    bool extract_feature_len) {
  const static std::unordered_set<std::string> unsupported{};
  Initialize(info, extract_feature_len);

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
      } else if (op.type() == "Int8Quantize") {
        InferInt8QuantizeInput(op);
      } else if (op.type() == "Mul" || op.type() == "Add") {
        InferElementwiseOpInput(op);
      }
    }
    inferFinished = old_shape_num == shape_info_.size();
    VLOG(1) << "old shape info num: " << old_shape_num
            << ", new shape info num: " << shape_info_.size();
    old_shape_num = shape_info_.size();
  }

  // Make sure shape has name
  EnsureShapeNames(&shape_info_);
}

TensorShape& BoundShapeInferencer::SetTensorBoundShapeIfNotExist(
    const std::string& name,
    const std::vector<TensorBoundShape::DimType>& t,
    std::vector<int64_t> bound_dims,
    TensorProto::DataType type,
    bool is_quantized) {
  return CheckAndSetTensorBoundShape(
      name, t, bound_dims, type, is_quantized, true);
}

// if allow_existing_shape is true, we use existing shape directly
// and not enforce shape to be equal to bound_dims
// else we enforce them to be equal
TensorShape& BoundShapeInferencer::CheckAndSetTensorBoundShape(
    const std::string& name,
    const std::vector<TensorBoundShape::DimType>& t,
    std::vector<int64_t> bound_dims,
    TensorProto::DataType type,
    bool is_quantized,
    bool allow_existing_shape,
    float scale,
    int offset,
    bool in_place_op) {
  auto rt = shape_info_.emplace(name, ShapeInfo());
  ShapeInfo& shape_info = rt.first->second;
  TensorShape& shape = shape_info.shape;
  if (shape_info.getShapeIsFinal()) {
    return shape;
  }
  if (is_quantized) {
    shape_info.is_quantized = true;
    shape_info.q_info.scale.clear();
    shape_info.q_info.scale.push_back(scale);
    shape_info.q_info.offset.clear();
    shape_info.q_info.offset.push_back(offset);
    shape_info.q_info.axis = 1;
  }
  // If the shape information exists in shape_info_ already and we want to
  // compare old/new shapes
  if (!rt.second && !in_place_op) {
    // Check dim size consistency
    CAFFE_ENFORCE_EQ(
        shape.dims_size(),
        bound_dims.size(),
        "Dim size inconsistency found in tensor ",
        name);
    // Get precedence of previous shape vs new shape
    int precedence = 0;
    if (!shape_info.dimTypeIsSet()) {
      precedence = 1;
    } else {
      precedence = takePrecedenceOver(shape_info.getDimType(), t);
    }
    // If precedence == 0: check whether previous shape == new shape
    // If precedence == 1, override shape with new value
    // If precedence == -1, previous shape takes precedence and
    // new value is skipped.
    if (precedence == 1) {
      shape_info.setDimType(t);
      for (auto i: c10::irange(bound_dims.size())) {
        shape.set_dims(i, bound_dims[i]);
      }
    } else if (precedence == 0 && !allow_existing_shape) {
      // Enforce previous dims and current dims are the same.
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
  // If shape information does not exist in shape_info_,
  // or shape info is not final,
  // set shape info according to inputs.
  if (!shape_info.getShapeIsFinal()) {
    shape_info.setDimType(t);
    shape.mutable_dims()->Clear();
    for (const auto d : bound_dims) {
      shape.add_dims(d);
    }
    shape.set_data_type(type);
    if (in_place_op) {
      shape_info.setShapeIsFinal(true);
    }
  }
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
    it->second.setDimType(std::vector<TensorBoundShape::DimType>(
        it->second.shape.dims_size(), TensorBoundShape_DimType_CONSTANT));
    if (op.type() == "ConstantFill" && op.input_size() >= 1) {
      auto it_input = shape_info_.find(op.input(0));
      if (it_input != shape_info_.end()) {
        it->second.setDimType(it_input->second.getDimType());
      }
    }
  }
}

void BoundShapeInferencer::InferLengthsRangeFill(const OperatorDef& op) {
  CAFFE_ENFORCE_EQ(op.input_size(), 1, "LengthsRangeFill must have 1 input");
  CAFFE_ENFORCE_EQ(op.output_size(), 1, "LengthsRangeFill must have 1 output");
  // Both input and ouptut of LengthsRangeFill is int32:
  // https://fburl.com/fhwb5666
  CheckAndSetTensorBoundShape(
      op.input(0),
      {TensorBoundShape_DimType_BATCH},
      {spec_.max_batch_size},
      TensorProto_DataType_INT32,
      false);
  CheckAndSetTensorBoundShape(
      op.output(0),
      {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
      {spec_.max_batch_size * spec_.max_seq_size},
      TensorProto_DataType_INT32,
      false);
  current_dim_type_ = TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT;
}

void BoundShapeInferencer::InferSparseLengthsSumSparseLookup(
    const OperatorDef& op) {
  CAFFE_ENFORCE_GT(
      op.input_size(),
      2,
      "SparseLengthsSumSparseLookup must have more than 2 input");
  CAFFE_ENFORCE_GT(
      op.output_size(),
      1,
      "SparseLengthsSumSparseLookup must have more than 1 output");
  if (shape_info_.find(op.input(2)) != shape_info_.end()) {
    LOG(WARNING)
        << "Shape of COMPRESSED_INDICES_MAPPING input of SparseLengthsSumSparseLookup "
        << op.input(2) << " needs to be presented";
  }
  for (int i = 0; i < 2; ++i) {
    const auto it = shape_info_.find(op.input(i));
    if (it != shape_info_.end()) {
      shape_info_[op.output(i)] = it->second;
    }
  }
  // Handle the weights
  if (op.input_size() == 4) {
    CAFFE_ENFORCE_EQ(op.output_size(), 3);
    const auto it = shape_info_.find(op.input(3));
    if (it != shape_info_.end()) {
      shape_info_[op.output(2)] = it->second;
    }
  }
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

  const int weight =
      (op.type() == "SparseLengthsWeightedSum" ||
       op.type() == "SparseLengthsWeightedSumFused8BitRowwise" ||
       op.type() == "SparseLengthsWeightedSumFused4BitRowwise" ||
       op.type() == "SparseLengthsWeightedSum4BitRowwiseSparse" ||
       op.type() == "SparseLengthsWeightedSum8BitRowwiseSparse")
      ? 1
      : 0;

  const bool is4bit =
      (op.type() == "SparseLengthsSumFused4BitRowwise" ||
       op.type() == "SparseLengthsWeightedSumFused4BitRowwise" ||
       op.type() == "SparseLengthsWeightedSum4BitRowwiseSparse" ||
       op.type() == "SparseLengthsSum4BitRowwiseSparse");

  if (weight) {
    CAFFE_ENFORCE_GE(
        op.input_size(),
        4,
        "SparseLengthsWeightedSum(Sparse) must have 4 or 5 inputs");
    SetTensorBoundShapeIfNotExist(
        op.input(weight),
        {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
        {spec_.max_batch_size * spec_.max_seq_size},
        TensorProto_DataType_FLOAT,
        false);
  }

  // Bound inputs
  SetTensorBoundShapeIfNotExist(
      op.input(1 + weight),
      {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
      {spec_.max_batch_size * spec_.max_seq_size},
      TensorProto_DataType_INT64,
      false);
  CheckAndSetTensorBoundShape(
      op.input(2 + weight),
      {TensorBoundShape_DimType_BATCH},
      {spec_.max_batch_size},
      TensorProto_DataType_INT32,
      false);

  // Infer output
  CAFFE_ENFORCE_EQ(it->second.shape.dims_size(), 2);
  current_dim_type_ = TensorBoundShape_DimType_BATCH;
  current_max_batch_size_ = spec_.max_batch_size;
  auto output_dim1 = it->second.shape.dims(1);
  // If the op is SparseLengthsSumFused8BitRowwise, we need to extract 4 bytes
  // for fp32 scale and 4 bytes for fp32 bias (https://fburl.com/t6dp9tsc)
  if (op.type() == "SparseLengthsSumFused8BitRowwise" ||
      op.type() == "SparseLengthsWeightedSumFused8BitRowwise" ||
      op.type() == "SparseLengthsSum8BitRowwiseSparse" ||
      op.type() == "SparseLengthsWeightedSum8BitRowwiseSparse") {
    output_dim1 -= 8;
  }
  // If the op is SparseLengthsSumFused4BitRowwise, we need to extract 2 bytes
  // for fp16 scale and 2 bytes for fp16 bias. Then we double it because we
  // pack 2 entries into 1 uint8 element of the embedding table.
  // (https://fburl.com/diffusion/stmsyz74)
  else if (is4bit) {
    output_dim1 -= 4;
    output_dim1 *= 2;
  }
  CAFFE_ENFORCE_GE(
      it->second.getDimType().size(), 2, "input(0): ", op.input(0));
  CheckAndSetTensorBoundShape(
      op.output(0),
      {TensorBoundShape_DimType_BATCH, it->second.getDimType(1)},
      {spec_.max_batch_size, output_dim1},
      TensorProto_DataType_FLOAT,
      false);
}

void BoundShapeInferencer::InferShape(const OperatorDef& op) {
  InferCommonOp(op);
  // old_shape should be a constant
  if (op.output_size() > 0 && shape_info_.count(op.output(0))) {
    shape_info_[op.output(0)].setDimType(0, TensorBoundShape_DimType_CONSTANT);
  }
}

void BoundShapeInferencer::InferReshape(const OperatorDef& op) {
  InferCommonOp(op);
  // old_shape should be a constant
  if (op.output_size() > 1 && shape_info_.count(op.output(1))) {
    shape_info_[op.output(1)].setDimType(0, TensorBoundShape_DimType_CONSTANT);
  }
}

void BoundShapeInferencer::InferInt8QuantizeInput(const OperatorDef& op) {
  if (op.output_size() == 0 || op.input_size() == 0) {
    return;
  }
  if (shape_info_.find(op.input(0)) != shape_info_.end()) {
    return;
  }
  const auto it = shape_info_.find(op.output(0));
  if (it == shape_info_.end()) {
    return;
  }
  auto input_shape_info = it->second;
  input_shape_info.is_quantized = false;
  input_shape_info.q_info.offset.clear();
  input_shape_info.q_info.scale.clear();
  input_shape_info.shape.set_data_type(TensorProto_DataType_FLOAT);
  shape_info_.emplace(op.input(0), std::move(input_shape_info));
}

void BoundShapeInferencer::InferElementwiseOpInput(const OperatorDef& op) {
  if (shape_info_.find(op.input(0)) != shape_info_.end() &&
      shape_info_.find(op.input(1)) != shape_info_.end()) {
    return;
  }
  const auto it = shape_info_.find(op.output(0));
  if (it == shape_info_.end()) {
    return;
  }
  ArgumentHelper helper(op);
  const bool broadcast = helper.GetSingleArgument<bool>("broadcast", false);
  if (broadcast) {
    auto input_shape_info = it->second;
    shape_info_.emplace(op.input(0), input_shape_info);
    // From definition of Add/Mul:
    // "When broadcasting is specified,
    // the second tensor can either be of size 1 (a scalar value),
    // or having its shape as a contiguous subset of the first tensors shape."
    // shape info of second input is always subset of first input.
    // Set bound shape of second input same as first input.
    shape_info_.emplace(op.input(1), std::move(input_shape_info));
  }
}

void BoundShapeInferencer::InferConcatInputs(const OperatorDef& op) {
  ArgumentHelper helper(op);
  const auto add_axis = helper.GetSingleArgument<int32_t>("add_axis", 0);
  // NOLINTNEXTLINE(bugprone-branch-clone)
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
      shape_info_[op.output(1)].setDimType(
          0, TensorBoundShape_DimType_CONSTANT);
    }
  }
}

void BoundShapeInferencer::InferElementwiseOp(const OperatorDef& op) {
  InferCommonOp(op);
  if (shape_info_.find(op.output(0)) != shape_info_.end() &&
      shape_info_.find(op.input(1)) != shape_info_.end()) {
    return;
  }
  const auto it = shape_info_.find(op.input(0));
  if (it == shape_info_.end()) {
    return;
  }
  ArgumentHelper helper(op);
  const bool broadcast = helper.GetSingleArgument<bool>("broadcast", false);
  if (broadcast) {
    auto input_shape_info = it->second;
    shape_info_.emplace(op.input(1), input_shape_info);
    shape_info_.emplace(op.output(0), std::move(input_shape_info));
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
      current_dim_type_ = ref_input_shape->getDimType(0);
      for (const auto& i : missing_shape_inputs) {
        shape_info_.emplace(i, *ref_input_shape);
      }
    }
  }
  InferCommonOp(op);
  // split_info should be a constant
  if (op.output_size() > 1 && shape_info_.count(op.output(1))) {
    shape_info_[op.output(1)].setDimType(0, TensorBoundShape_DimType_CONSTANT);
  }
}

void BoundShapeInferencer::InferFC(const OperatorDef& op) {
  CAFFE_ENFORCE(
      op.input_size() == 3 || op.input_size() == 4,
      "FC has to have 3 or 4 inputs");
  const auto w_it = shape_info_.find(op.input(1));
  CAFFE_ENFORCE(
      w_it != shape_info_.end(),
      "Shape of WEIGHT input of FC ",
      op.input(1),
      " needs to be presented");
  const ShapeInfo& w_shape_info = w_it->second;
  const auto b_it = shape_info_.find(op.input(2));
  CAFFE_ENFORCE(
      b_it != shape_info_.end(),
      "Shape of BIAS input of FC ",
      op.input(2),
      " needs to be presented");
  const ShapeInfo& b_shape_info = b_it->second;
  bool fp16 = (op.type() == "FbFCPacked");
  bool int8_fc = (op.type() == "Int8FC" || op.engine() == "DNNLOWP");
  float scale = 1;
  int offset = 0;

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
    std::vector<TensorBoundShape::DimType> dimTypes;
    for (int i = 0; i < axis - 1; ++i) {
      dims.push_back(1);
      dimTypes.push_back(TensorBoundShape_DimType_CONSTANT);
    }
    dims.push_back(spec_.max_batch_size);
    dimTypes.push_back(TensorBoundShape_DimType_BATCH);
    dims.push_back(K);
    dimTypes.push_back(TensorBoundShape_DimType_CONSTANT);
    current_dim_type_ = TensorBoundShape_DimType_BATCH;
    current_max_batch_size_ = spec_.max_batch_size;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    TensorProto::DataType w_data_type;
    if (fp16) {
      w_data_type = TensorProto_DataType_FLOAT;
    } else if (int8_fc) {
      w_data_type = TensorProto_DataType_UINT8;
    } else {
      w_data_type = w_shape.data_type();
    }

    if (int8_fc) {
      scale = helper.GetSingleArgument<float>("Y_scale", 1);
      offset = helper.GetSingleArgument<int>("Y_zero_point", 0);
    }
    // Note: for FbFCPacked, weight is fp16 but activations are in fp32
    CheckAndSetTensorBoundShape(
        op.input(0),
        dimTypes,
        dims,
        w_data_type,
        int8_fc ? true : false,
        false,
        scale,
        offset);
  } else {
    ShapeInfo& x_shape_info = x_it->second;
    if (x_shape_info.getDimType(0) == TensorBoundShape_DimType_UNKNOWN) {
      CAFFE_ENFORCE_GE(x_shape_info.shape.dims_size(), 1);
      x_shape_info.shape.set_dims(0, spec_.max_batch_size);
      x_shape_info.setDimType(0, TensorBoundShape_DimType_BATCH);
    }
  }

  // Standard shape inference for outputs
  std::vector<TensorShape> input_shapes{
      shape_info_[op.input(0)].shape, w_shape_info.shape, b_shape_info.shape};
  if (op.input_size() == 4) {
    const auto quant_param_it = shape_info_.find(op.input(3));
    CAFFE_ENFORCE(
        quant_param_it != shape_info_.end(),
        "Shape of quant_param input of FC ",
        op.input(3),
        " needs to be presented");
    const ShapeInfo& quant_param_shape_info = quant_param_it->second;
    input_shapes.emplace_back(quant_param_shape_info.shape);
  }
  std::vector<TensorShape> output_shapes = InferOutput(op, input_shapes);
  CAFFE_ENFORCE_EQ(output_shapes.size(), 1);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  TensorProto::DataType output_data_type;
  if (fp16) {
    output_data_type = TensorProto_DataType_FLOAT;
  } else if (int8_fc) {
    output_data_type = TensorProto_DataType_UINT8;
  } else {
    output_data_type = output_shapes.front().data_type();
  }

  if (int8_fc) {
    ArgumentHelper helper(op);

    scale = helper.GetSingleArgument<float>("Y_scale", 1);
    offset = helper.GetSingleArgument<int>("Y_zero_point", 0);
  }

  CheckAndSetTensorBoundShape(
      op.output(0),
      setDimTypeWithFirst(
          TensorBoundShape_DimType_BATCH, output_shapes.front().dims().size()),
      ConvertToVec(output_shapes[0].dims()),
      output_data_type,
      int8_fc ? true : false,
      false,
      scale,
      offset);
}

// Infers shapes for operators which are used to transform non-quantized
// operators (e.g. SparseLengthsSum) into quantized operators (e.g.
// SparseLengthsSumFused8BitRowwise) at model training time. If we're doing
// quantization for CONSTANTS (eg. embedding tables), current_dim_type_ should
// be set to CONSTANT.
void BoundShapeInferencer::InferQuantizationTransformation(
    const OperatorDef& op) {
  bool all_constant = true;
  for (const auto& input : op.input()) {
    const auto it = shape_info_.find(input);
    if (it == shape_info_.end() ||
        it->second.getDimType(0) != TensorBoundShape_DimType_CONSTANT) {
      all_constant = false;
      break;
    }
  }
  const auto previous_dim_type = current_dim_type_;
  if (all_constant) {
    current_dim_type_ = TensorBoundShape_DimType_CONSTANT;
  }
  InferCommonOp(op);
  current_dim_type_ = previous_dim_type;
}

void BoundShapeInferencer::InferUnPackRecords(const OperatorDef& op) {
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

  std::vector<TensorShape> output_shapes;

  ArgumentHelper helper(op);
  std::vector<std::string> fields =
      helper.GetRepeatedArgument<std::string>("fields");

  const int num_tensors = fields.size();
  if (spec_.max_batch_size == 1 && num_tensors == 1 &&
      input_shapes[0].dims_size() != 1) {
    // Special case of single tensor input
    output_shapes.push_back(input_shapes[0]);
  } else {
    // Input is packed
    TensorShape oshape;
    oshape.add_dims(spec_.max_batch_size);
    oshape.add_dims(spec_.num_embeddings);
    oshape.add_dims(spec_.embedding_length);
    // TODO: how to do this more intelligently
    oshape.set_data_type(TensorProto::FLOAT);
    for (int i = 0; i < num_tensors; i++) {
      output_shapes.push_back(oshape);
    }
  }

  for (auto i: c10::irange(output_shapes.size())) {
    const auto& shape = output_shapes[i];

    CheckAndSetTensorBoundShape(
        op.output(i),
        setDimTypeWithFirst(current_dim_type_, shape.dims().size()),
        ConvertToVec(shape.dims()),
        output_shapes[i].data_type(),
        false);
  }
}

void BoundShapeInferencer::InferTile(const OperatorDef& op) {
  if (op.input_size() > 1) {
    LOG(WARNING) << "Cannot infer shape for Tile when axis and tils are inputs";
    return;
  }
  const auto it = shape_info_.find(op.input(0));
  if (it == shape_info_.end()) {
    LOG(WARNING) << "Cannot find shape info for " << op.input(0)
                 << ". Skipping " << op.type();
    return;
  }

  ArgumentHelper helper(op);
  const std::int32_t tiles = helper.GetSingleArgument<std::int32_t>("tiles", 1);
  std::int32_t axis = helper.GetSingleArgument<std::int32_t>("axis", 0);
  bool dynamic = helper.GetSingleArgument<bool>("dynamic", false);
  auto ndims = it->second.shape.dims_size();
  const auto canonical_axis = canonical_axis_index_(axis, ndims);
  auto shape = it->second.shape;
  shape.set_dims(
      canonical_axis,
      shape.dims(canonical_axis) * (dynamic ? spec_.max_batch_size : tiles));
  CheckAndSetTensorBoundShape(
      op.output(0),
      setDimTypeWithFirst(TensorBoundShape_DimType_BATCH, ndims),
      ConvertToVec(shape.dims()),
      it->second.shape.data_type(),
      false);
}

void BoundShapeInferencer::InferSoftmax(const OperatorDef& op) {
  CAFFE_ENFORCE_EQ(op.input_size(), 1, op.type(), " must have 1 input");
  CAFFE_ENFORCE_EQ(op.output_size(), 1, op.type(), " must have 1 output");

  auto it = shape_info_.find(op.input(0));
  if (it == shape_info_.end()) {
    LOG(WARNING) << "Didn't find shape info for the input of Softmax, skipping";
    return;
  }

  CheckAndSetTensorBoundShape(
      op.output(0),
      setDimTypeWithFirst(it->second.getDimType(0), it->second.shape.dims_size()),
      ConvertToVec(it->second.shape.dims()),
      it->second.shape.data_type(),
      false);
}

void BoundShapeInferencer::InferBucketize(const OperatorDef& op) {
  CAFFE_ENFORCE_EQ(op.input_size(), 1, op.type(), " must have 1 input");
  CAFFE_ENFORCE_EQ(op.output_size(), 1, op.type(), " must have 1 output");

  auto it = shape_info_.find(op.input(0));
  if (it == shape_info_.end()) {
    LOG(WARNING) << "Didn't find shape info for the input of Bucketize, skipping";
    return;
  }

  InferCommonOp(op);
  auto it_output = shape_info_.find(op.output(0));
  if (it_output != shape_info_.end()) {
    it_output->second.setDimType(it->second.getDimType());
  }
}

void BoundShapeInferencer::InferLpNorm(const OperatorDef& op) {
  CAFFE_ENFORCE_EQ(op.output_size(), 1, op.type(), " must have 1 output");
  InferCommonOp(op);
  auto it = shape_info_.find(op.output(0));
  if (it != shape_info_.end()) {
    it->second.setDimType(std::vector<TensorBoundShape::DimType>(
        it->second.shape.dims_size(), TensorBoundShape_DimType_CONSTANT));
  }
}

void BoundShapeInferencer::InferClip(const OperatorDef& op) {
  CAFFE_ENFORCE_EQ(op.output_size(), 1, op.type(), " must have 1 output");
  InferCommonOp(op);
  auto it = shape_info_.find(op.output(0));
  if (it != shape_info_.end()) {
    auto it_input = shape_info_.find(op.input(0));
    if (it_input != shape_info_.end()) {
      it->second.setDimType(it_input->second.getDimType());
    }
  }
}

void BoundShapeInferencer::InferMean(const OperatorDef& op) {
  CAFFE_ENFORCE_EQ(op.output_size(), 1, op.type(), " must have at 1 output");
  InferCommonOp(op);
  auto it = shape_info_.find(op.output(0));
  if (it != shape_info_.end()) {
    auto it_input = shape_info_.find(op.input(0));
    if (it_input != shape_info_.end()) {
      it->second.setDimType(it_input->second.getDimType());
    }
  }
}

void BoundShapeInferencer::InferDiv(const OperatorDef& op) {
  CAFFE_ENFORCE_EQ(op.output_size(), 1, op.type(), " must have 1 output");
  InferCommonOp(op);
  auto it = shape_info_.find(op.output(0));
  if (it != shape_info_.end()) {
    auto it_input = shape_info_.find(op.input(0));
    if (it_input != shape_info_.end()) {
      it->second.setDimType(it_input->second.getDimType());
    }
  }
}

void BoundShapeInferencer::InferTranspose(const OperatorDef& op) {
  CAFFE_ENFORCE_EQ(op.input_size(), 1, op.type(), " must have 1 input");
  CAFFE_ENFORCE_EQ(op.output_size(), 1, op.type(), " must have 1 output");

  auto it = shape_info_.find(op.input(0));
  if (it == shape_info_.end()) {
    LOG(WARNING) << "Didn't find shape info for the input of Transpose";
    return;
  }

  ArgumentHelper helper(op);
  std::vector<int> axes = helper.GetRepeatedArgument<int>("axes");
  if (axes.empty()) {
    // In this case it should be existing dims in reverse order
    for (int i = it->second.shape.dims().size() - 1; i >= 0; --i) {
      axes.push_back(i);
    }
  } else {
    CAFFE_ENFORCE_EQ(
        axes.size(),
        it->second.shape.dims().size(),
        op.type(),
        " must specify all axes in Transpose."
    );
    auto valid_axes =
        std::all_of(axes.begin(), axes.end(), [numDims = it->second.shape.dims().size()](int& axis) {
          return axis >= 0 && axis < numDims;
        });
    CAFFE_ENFORCE(valid_axes, "Invalid axes were provided.");
  }

  std::vector<TensorBoundShape::DimType> dimTypes;
  std::vector<int64_t> dims;
  for (auto axis : axes) {
    dimTypes.push_back(it->second.getDimType(axis));
    dims.push_back(it->second.shape.dims()[axis]);
  }

  CheckAndSetTensorBoundShape(
      op.output(0),
      dimTypes,
      dims,
      it->second.shape.data_type(),
      false);
}

void BoundShapeInferencer::InferCommonOp(
    const OperatorDef& op,
    const OpSchema* schema,
    bool bypass_input_check,
    bool in_place_op) {
  // First, we need to check that all the input shape/types are already
  // presented
  try {
    const static std::unordered_set<std::string>
        types_with_independent_output_shape = {
            "Int8GenQuantParams",
            "Int8QuantSchemeBlobFill",
            "ComputeEqualizationScale",
            "Int8GenQuantParamsMinMax"};
    std::vector<TensorShape> input_shapes;
    for (const auto& input : op.input()) {
      const auto it = shape_info_.find(input);
      if (it == shape_info_.end() &&
          !types_with_independent_output_shape.count(op.type()) &&
          !bypass_input_check) {
        LOG(WARNING) << "Cannot find shape info for " << input << ". Skipping "
                     << op.type();
        return;
      }
      if (types_with_independent_output_shape.count(op.type()) ||
          (bypass_input_check && it == shape_info_.end())) {
        TensorShape input_shape;
        input_shapes.emplace_back(std::move(input_shape));
      } else {
        input_shapes.emplace_back(it->second.shape);
      }
    }

    // Schema can be pre-defined.
    // If not predefined, get the schema for the op.
    if (schema == nullptr) {
      schema = OpSchemaRegistry::Schema(op.type());
    }
    CAFFE_ENFORCE(schema);
    std::vector<TensorShape> output_shapes;
    output_shapes = schema->InferTensor(op, input_shapes);
    bool is_quantized = !(op.type().compare(0, 4, "Int8")) &&
        (op.type() != "Int8Dequantize") &&
        (op.type() != "Int8QuantSchemeBlobFill") &&
        (op.type() != "ComputeEqualizationScale") &&
        (op.type() != "Int8GenQuantParams") &&
        (op.type() != "Int8GenQuantParamsMinMax");
    float scale = 1;
    int offset = 0;

    TensorProto::DataType infered_data_type = TensorProto::UNDEFINED;
    if (is_quantized) {
      const static std::map<std::string, int> type_info_from_input = {
          {"Int8Quantize", -1}, // Force this op's output to be uint8
          {"Int8FCPackWeight", 0},
          {"Int8ConvPackWeight", 0},
          {"Int8ConvRelu", 1},
          {"Int8MaxPool", 0},
          {"Int8AveragePool", 0},
          {"Int8FC", 1},
          {"Int8Conv", 1},
          {"Int8SumRelu", 0},
          {"Int8Relu", 0}};
      CAFFE_ENFORCE(
          type_info_from_input.find(op.type()) != type_info_from_input.end(),
          "Undefined quantized output data type, add it into type_info_from_input");
      int target = type_info_from_input.find(op.type())->second;
      if (target == -1) {
        infered_data_type = TensorProto::UINT8;
      } else {
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        CAFFE_ENFORCE(target < input_shapes.size());
        infered_data_type = input_shapes[target].data_type();
      }

      // Extract output scale and offset
      ArgumentHelper helper(op);
      scale = helper.GetSingleArgument<float>("Y_scale", 1);
      offset = helper.GetSingleArgument<int>("Y_zero_point", 0);
    } else if (op.type() == "Int8Dequantize") {
      infered_data_type = TensorProto::FLOAT;
    }

    for (auto i: c10::irange(output_shapes.size())) {
      const auto& shape = output_shapes[i];
      if (shape.unknown_shape()) {
        continue;
      }
      auto tmp_dtype = infered_data_type;
      if (infered_data_type == TensorProto::UNDEFINED) {
        infered_data_type = shape.data_type();
      }
      CheckAndSetTensorBoundShape(
          op.output(i),
          setDimTypeWithFirst(current_dim_type_, shape.dims().size()),
          ConvertToVec(shape.dims()),
          infered_data_type,
          is_quantized,
          false,
          scale,
          offset,
          in_place_op);
      infered_data_type = tmp_dtype;
    }
  } catch (const caffe2::EnforceNotMet& e) {
    LOG(ERROR) << "Enforce not met while inferring shapes for " << op.type()
               << ": " << e.what() << " first output: " << op.output(0);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Caught exception while inferring shapes for " << op.type()
                 << ": " << e.what() << " first output: " << op.output(0);
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
