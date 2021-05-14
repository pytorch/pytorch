#include "caffe2/opt/shape_info.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/utils/string_utils.h"
#include <cctype>

namespace caffe2 {

namespace {
bool isNumber(const std::string& s) {
  bool empty = true;
  for (const char c : s) {
    if (std::isalpha(c)) {
      return false;
    }
    if (!std::isspace(c)) {
      empty = false;
    }
  }
  return !empty;
}

std::string toLower(const std::string& s) {
  std::string t;
  t.resize(s.size());
  for (size_t i = 0; i < t.size(); i++) {
    t[i] = std::tolower(s[i]);
  }
  return t;
}

TensorProto_DataType toTensorProtoDataType(const std::string& in) {
  std::string s = toLower(in);
  if (s == "uint8") {
    return TensorProto_DataType_UINT8;
  } else if (s == "int8") {
    return TensorProto_DataType_INT8;
  } else if (s == "uint16") {
    return TensorProto_DataType_UINT16;
  } else if (s == "int16") {
    return TensorProto_DataType_INT16;
  } else if (s == "int32") {
    return TensorProto_DataType_INT32;
  } else if (s == "int64") {
    return TensorProto_DataType_INT64;
  } else if (s == "float16" || s == "half") {
    return TensorProto_DataType_FLOAT16;
  } else if (s == "float") {
    return TensorProto_DataType_FLOAT;
  } else if (s == "double") {
    return TensorProto_DataType_DOUBLE;
  } else if (s == "byte") {
    return TensorProto_DataType_BYTE;
  } else if (s == "string") {
    return TensorProto_DataType_STRING;
  } else if (s == "bool") {
    return TensorProto_DataType_BOOL;
  } else if (s == "hash") {
    return TensorProto_DataType_ZERO_COLLISION_HASH;
  }
  // return default data type, float
  return TensorProto_DataType_FLOAT;
}
} // namespace

ShapeInfo getShapeInfoFromBlob(const Blob* blob) {
  ShapeInfo shape_info;
  shape_info.shape = GetTensorShapeOfBlob(blob);
  if (!shape_info.shape.unknown_shape()) {
    shape_info.setDimType(std::vector<TensorBoundShape::DimType>(
        shape_info.shape.dims_size(), TensorBoundShape_DimType_CONSTANT));
  }
  if (blob->meta().id() == TypeMeta::Id<int8::Int8TensorCPU>()) {
    shape_info.is_quantized = true;
    LoadInt8TensorInfoOfBlob(
        &shape_info.q_info.scale,
        &shape_info.q_info.offset,
        &shape_info.q_info.axis,
        blob);
  } else {
#ifndef C10_MOBILE
    auto function_ptr =
        ExternalTensorFunctionsBaseRegistry()->Create(blob->meta().id());
    if (function_ptr != nullptr) {
      shape_info.is_quantized = function_ptr->isQuantized();
      function_ptr->LoadInfoOfBlob(
          blob,
          &shape_info.q_info.scale,
          &shape_info.q_info.offset,
          &shape_info.q_info.axis);
    }
#endif
  }
  return shape_info;
}

void modifyTensorShapeDimSize(
    TensorShape* tensor_shape,
    int dim_index,
    const int64_t old_size,
    const int64_t new_size) {
  CAFFE_ENFORCE(
      old_size > 0, "Old size should be non-zero, old_size: ", old_size);
  CAFFE_ENFORCE(
      tensor_shape->dims(dim_index) % old_size == 0,
      "tensor_shape->dims[",
      dim_index,
      "] = ",
      tensor_shape->dims(dim_index),
      " cannot be divided by old_size ",
      old_size,
      " tensor name: ",
      tensor_shape->name());
  int64_t modified_size = (tensor_shape->dims(dim_index) * new_size) / old_size;
  tensor_shape->set_dims(dim_index, modified_size);
}

void changeTensorBoundShapes(
    TensorBoundShape& tensor_shape_and_type,
    const int64_t old_batch_size,
    const int64_t old_seq_size,
    const int64_t new_batch_size,
    const int64_t new_seq_size) {
  CAFFE_ENFORCE(
      tensor_shape_and_type.dim_type().size() ==
      tensor_shape_and_type.shape().dims().size());

  for (int i = 0; i < tensor_shape_and_type.dim_type().size(); i++) {
    TensorBoundShape_DimType dim_type = tensor_shape_and_type.dim_type(i);
    // Need to change max_batch_size
    if (dim_type == TensorBoundShape_DimType_BATCH ||
        dim_type == TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX ||
        dim_type == TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT) {
      TensorShape* tensor_shape = tensor_shape_and_type.mutable_shape();
      modifyTensorShapeDimSize(tensor_shape, i, old_batch_size, new_batch_size);
    }
    // Need to change max_seq_size
    if (dim_type == TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT ||
        dim_type == TensorBoundShape_DimType_FEATURE_MAX_DEFAULT) {
      TensorShape* tensor_shape = tensor_shape_and_type.mutable_shape();
      modifyTensorShapeDimSize(tensor_shape, i, old_seq_size, new_seq_size);
    }
  }
}

ShapeInfoMap extractShapeInfoFromTensorBoundShapes(
    TensorBoundShapes tensor_bound_shapes,
    int64_t new_max_batch_size,
    int64_t new_max_feature_len) {
  ShapeInfoMap shape_info_map;
  if (new_max_batch_size == -1) {
    new_max_batch_size = tensor_bound_shapes.max_batch_size();
  }
  if (new_max_feature_len == -1) {
    new_max_feature_len = tensor_bound_shapes.max_feature_len();
  }
  for (auto& tensor_bound_shape : *(tensor_bound_shapes.mutable_shapes())) {
    std::vector<TensorBoundShape::DimType> dim_types;
    dim_types.reserve(tensor_bound_shape.shape().dims_size());
    for (auto dim_type : tensor_bound_shape.dim_type()) {
      dim_types.emplace_back(TensorBoundShape::DimType(dim_type));
    }
    changeTensorBoundShapes(
        tensor_bound_shape,
        tensor_bound_shapes.max_batch_size(),
        tensor_bound_shapes.max_feature_len(),
        new_max_batch_size,
        new_max_feature_len);
    shape_info_map[tensor_bound_shape.name()] =
        // NOLINTNEXTLINE(performance-move-const-arg)
        ShapeInfo(dim_types, std::move(tensor_bound_shape.shape()));
  }
  return shape_info_map;
}

bool operator==(const ShapeInfo& lhs, const ShapeInfo& rhs) {
  return lhs.getDimType() == rhs.getDimType() &&
      lhs.shape.SerializeAsString() == rhs.shape.SerializeAsString();
}

ShapeInfo constructShapeInfoWithDefaultDimType(
    TensorShape shape,
    TensorBoundShape_DimType defaultFirstDimType) {
  std::vector<TensorBoundShape_DimType> dimType(
      shape.dims_size(), TensorBoundShape_DimType_CONSTANT);
  if (dimType.size()) {
    dimType[0] = defaultFirstDimType;
  }
  return ShapeInfo(dimType, shape);
}

void parseShapeInfoMapFromString(
    const std::string& input,
    ShapeInfoMap& shape_hints) {
  auto hints = caffe2::split('#', input);
  for (const auto& hint : hints) {
    auto kv = caffe2::split(',', hint);
    CAFFE_ENFORCE_GE(kv.size(), 2, "Cannot parse shape hint: ", hint);
    const auto& name = kv[0];

    TensorShape shape;
    size_t size = kv.size();
    CAFFE_ENFORCE_GT(size, 1);
    if (!isNumber(kv[size - 1])) {
      // last value is the type
      shape.set_data_type(toTensorProtoDataType(kv[size - 1]));
      size--;
    } else {
      if (name.find("int8") != std::string::npos) {
        // Kept for backwards compatibility.
        // Set type explicitly to overwrite it.
        shape.set_data_type(TensorProto_DataType_UINT8);
      } else {
        shape.set_data_type(TensorProto_DataType_FLOAT);
      }
    }

    bool valid = true;
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    for (int i = 1; i < size; i++) {
      auto dim = kv[i];
      try {
        shape.add_dims(std::stoi(dim));
      } catch (const std::exception& e) {
        valid = false;
        CAFFE_THROW("Cannot parse shape hint: ", hint);
      }
    }
    if (valid) {
      shape_hints.emplace(name, constructShapeInfoWithDefaultDimType(shape));
    }
  }
}

} // namespace caffe2
