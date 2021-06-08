#include "caffe2/core/types.h"
#include <c10/util/typeid.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace caffe2 {

TensorProto::DataType TypeMetaToDataType(const TypeMeta meta) {
  static_assert(
      sizeof(int) == 4, "int in this compiler does not equal to 4 bytes.");
  static std::map<TypeIdentifier, TensorProto::DataType> data_type_map{
      {TypeMeta::Id<float>(), TensorProto_DataType_FLOAT},
      {TypeMeta::Id<int>(), TensorProto_DataType_INT32},
      // BYTE does not have a type meta to proto mapping: we should
      // always use uint8_t when serializing. BYTE is kept for backward
      // compatibility.
      // {TypeMeta::Id<>(), TensorProto_DataType_BYTE},
      {TypeMeta::Id<string>(), TensorProto_DataType_STRING},
      {TypeMeta::Id<bool>(), TensorProto_DataType_BOOL},
      {TypeMeta::Id<uint8_t>(), TensorProto_DataType_UINT8},
      {TypeMeta::Id<int8_t>(), TensorProto_DataType_INT8},
      {TypeMeta::Id<uint16_t>(), TensorProto_DataType_UINT16},
      {TypeMeta::Id<int16_t>(), TensorProto_DataType_INT16},
      {TypeMeta::Id<int64_t>(), TensorProto_DataType_INT64},
      {TypeMeta::Id<at::Half>(), TensorProto_DataType_FLOAT16},
      {TypeMeta::Id<double>(), TensorProto_DataType_DOUBLE},
      {TypeMeta::Id<c10::qint8>(), TensorProto_DataType_INT8},
      {TypeMeta::Id<c10::quint8>(), TensorProto_DataType_UINT8},
      {TypeMeta::Id<c10::qint32>(), TensorProto_DataType_INT32},
  };
  const auto it = data_type_map.find(meta.id());
  return (
      it == data_type_map.end() ? TensorProto_DataType_UNDEFINED : it->second);
}

const TypeMeta DataTypeToTypeMeta(const TensorProto::DataType& dt) {
  static std::map<TensorProto::DataType, TypeMeta> type_meta_map{
      {TensorProto_DataType_FLOAT, TypeMeta::Make<float>()},
      {TensorProto_DataType_INT32, TypeMeta::Make<int>()},
      {TensorProto_DataType_BYTE, TypeMeta::Make<uint8_t>()},
      {TensorProto_DataType_STRING, TypeMeta::Make<std::string>()},
      {TensorProto_DataType_BOOL, TypeMeta::Make<bool>()},
      {TensorProto_DataType_UINT8, TypeMeta::Make<uint8_t>()},
      {TensorProto_DataType_INT8, TypeMeta::Make<int8_t>()},
      {TensorProto_DataType_UINT16, TypeMeta::Make<uint16_t>()},
      {TensorProto_DataType_INT16, TypeMeta::Make<int16_t>()},
      {TensorProto_DataType_INT64, TypeMeta::Make<int64_t>()},
      {TensorProto_DataType_FLOAT16, TypeMeta::Make<at::Half>()},
      {TensorProto_DataType_DOUBLE, TypeMeta::Make<double>()},
  };
  const auto it = type_meta_map.find(dt);
  if (it == type_meta_map.end()) {
    throw std::runtime_error("Unknown data type.");
  }
  return it->second;
}

} // namespace caffe2
