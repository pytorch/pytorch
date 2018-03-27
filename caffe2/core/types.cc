#include "caffe2/core/types.h"
#include "caffe2/core/typeid.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace caffe2 {

CAFFE_KNOWN_TYPE(float);
CAFFE_KNOWN_TYPE(int);
CAFFE_KNOWN_TYPE(std::string);
CAFFE_KNOWN_TYPE(bool);
CAFFE_KNOWN_TYPE(uint8_t);
CAFFE_KNOWN_TYPE(int8_t);
CAFFE_KNOWN_TYPE(uint16_t);
CAFFE_KNOWN_TYPE(int16_t);
CAFFE_KNOWN_TYPE(int64_t);
CAFFE_KNOWN_TYPE(float16);
CAFFE_KNOWN_TYPE(double);
CAFFE_KNOWN_TYPE(char);
CAFFE_KNOWN_TYPE(std::unique_ptr<std::mutex>);
CAFFE_KNOWN_TYPE(std::unique_ptr<std::atomic<bool>>);
CAFFE_KNOWN_TYPE(std::vector<int64_t>);
CAFFE_KNOWN_TYPE(std::vector<unsigned long>);
CAFFE_KNOWN_TYPE(bool*);
CAFFE_KNOWN_TYPE(char*);
CAFFE_KNOWN_TYPE(int*);

#ifdef CAFFE2_UNIQUE_LONG_TYPEMETA
CAFFE_KNOWN_TYPE(long);
CAFFE_KNOWN_TYPE(std::vector<long>);
#endif // CAFFE2_UNIQUE_LONG_TYPEMETA

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

const TypeMeta& DataTypeToTypeMeta(const TensorProto::DataType& dt) {
  static std::map<TensorProto::DataType, TypeMeta> type_meta_map{
      {TensorProto_DataType_FLOAT, TypeMeta::Make<float>()},
      {TensorProto_DataType_INT32, TypeMeta::Make<int>()},
      {TensorProto_DataType_STRING, TypeMeta::Make<std::string>()},
      {TensorProto_DataType_BOOL, TypeMeta::Make<bool>()},
      {TensorProto_DataType_UINT8, TypeMeta::Make<uint8_t>()},
      {TensorProto_DataType_INT8, TypeMeta::Make<int8_t>()},
      {TensorProto_DataType_UINT16, TypeMeta::Make<uint16_t>()},
      {TensorProto_DataType_INT16, TypeMeta::Make<int16_t>()},
      {TensorProto_DataType_INT64, TypeMeta::Make<int64_t>()},
      {TensorProto_DataType_FLOAT16, TypeMeta::Make<float16>()},
      {TensorProto_DataType_DOUBLE, TypeMeta::Make<double>()},
  };
  const auto it = type_meta_map.find(dt);
  if (it == type_meta_map.end()) {
    throw std::runtime_error("Unknown data type.");
  }
  return it->second;
}

}  // namespace caffe2
