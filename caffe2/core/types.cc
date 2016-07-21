#include "caffe2/core/types.h"
#include "caffe2/core/typeid.h"

namespace caffe2 {

TensorProto::DataType TypeMetaToDataType(const TypeMeta& meta) {
  static_assert(sizeof(int) == 4,
                "int in this compiler does not equal to 4 bytes.");
  static std::map<CaffeTypeId, TensorProto::DataType> data_type_map {
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
    {TypeMeta::Id<float16>(), TensorProto_DataType_FLOAT16},
    {TypeMeta::Id<double>(), TensorProto_DataType_DOUBLE},
  };
  const auto it = data_type_map.find(meta.id());
  return (it == data_type_map.end()
          ? TensorProto_DataType_UNDEFINED : it->second);
}

}  // namespace caffe2
