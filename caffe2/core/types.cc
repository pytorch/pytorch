#include "caffe2/core/types.h"
#include <c10/util/typeid.h>

#include <atomic>
#include <memory>
#include <string>

namespace caffe2 {

TensorProto::DataType TypeMetaToDataType(const TypeMeta& meta) {
  static_assert(
      sizeof(int) == 4, "int in this compiler does not equal to 4 bytes.");

  // Can't use a switch because `meta_id` is not an integer type
  const auto meta_id = meta.id();
  if (meta_id == TypeMeta::Id<float>()) {
    return TensorProto_DataType_FLOAT;
  } else if (meta_id == TypeMeta::Id<int>()) {
    return TensorProto_DataType_INT32;
  } else if (meta_id == TypeMeta::Id<string>()) {
    return TensorProto_DataType_STRING;
  } else if (meta_id == TypeMeta::Id<bool>()) {
    return TensorProto_DataType_BOOL;
  } else if (meta_id == TypeMeta::Id<uint8_t>()) {
    return TensorProto_DataType_UINT8;
  } else if (meta_id == TypeMeta::Id<int8_t>()) {
    return TensorProto_DataType_INT8;
  } else if (meta_id == TypeMeta::Id<uint16_t>()) {
    return TensorProto_DataType_UINT16;
  } else if (meta_id == TypeMeta::Id<int16_t>()) {
    return TensorProto_DataType_INT16;
  } else if (meta_id == TypeMeta::Id<int64_t>()) {
    return TensorProto_DataType_INT64;
  } else if (meta_id == TypeMeta::Id<at::Half>()) {
    return TensorProto_DataType_FLOAT16;
  } else if (meta_id == TypeMeta::Id<double>()) {
    return TensorProto_DataType_DOUBLE;
  } else if (meta_id == TypeMeta::Id<c10::qint8>()) {
    return TensorProto_DataType_INT8;
  } else if (meta_id == TypeMeta::Id<c10::quint8>()) {
    return TensorProto_DataType_UINT8;
  } else if (meta_id == TypeMeta::Id<c10::qint32>()) {
    return TensorProto_DataType_INT32;
  } else {
    return TensorProto_DataType_UNDEFINED;
  }
}

const TypeMeta DataTypeToTypeMeta(const TensorProto_DataType& dt) {
  switch (dt) {
    case TensorProto_DataType_FLOAT:
      return TypeMeta::Make<float>();
    case TensorProto_DataType_INT32:
      return TypeMeta::Make<int>();
    case TensorProto_DataType_BYTE:
      return TypeMeta::Make<uint8_t>();
    case TensorProto_DataType_STRING:
      return TypeMeta::Make<std::string>();
    case TensorProto_DataType_BOOL:
      return TypeMeta::Make<bool>();
    case TensorProto_DataType_UINT8:
      return TypeMeta::Make<uint8_t>();
    case TensorProto_DataType_INT8:
      return TypeMeta::Make<int8_t>();
    case TensorProto_DataType_UINT16:
      return TypeMeta::Make<uint16_t>();
    case TensorProto_DataType_INT16:
      return TypeMeta::Make<int16_t>();
    case TensorProto_DataType_INT64:
      return TypeMeta::Make<int64_t>();
    case TensorProto_DataType_FLOAT16:
      return TypeMeta::Make<at::Half>();
    case TensorProto_DataType_DOUBLE:
      return TypeMeta::Make<double>();
    default:
      throw std::runtime_error("Unknown data type.");
  };
}

} // namespace caffe2
