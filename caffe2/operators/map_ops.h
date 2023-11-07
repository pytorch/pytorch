#ifndef CAFFE2_OPERATORS_MAP_OPS_H_
#define CAFFE2_OPERATORS_MAP_OPS_H_

#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include <c10/util/irange.h>

#include <algorithm>
#include <iterator>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

namespace caffe2 {

template <typename T>
struct TypeNameTraits {
  static constexpr const char* name = "unknown";
};

template <>
struct TypeNameTraits<int64_t> {
  static constexpr const char* name = "int64_t";
};

template <>
struct TypeNameTraits<int32_t> {
  static constexpr const char* name = "int32_t";
};

template <typename KEY_T, typename VALUE_T>
struct MapTypeTraits {
  using MapType = std::unordered_map<KEY_T, VALUE_T>;
  static string MapTypeName() {
    return string("(std::unordered_map<") + TypeNameTraits<KEY_T>::name + ", " +
        TypeNameTraits<VALUE_T>::name + ">)";
  }
};

using MapType64To64 = MapTypeTraits<int64_t, int64_t>::MapType;
using MapType64To32 = MapTypeTraits<int64_t, int32_t>::MapType;
using MapType32To32 = MapTypeTraits<int32_t, int32_t>::MapType;
using MapType32To64 = MapTypeTraits<int32_t, int64_t>::MapType;

template <class Context>
class CreateMapOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit CreateMapOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  ~CreateMapOp() override {}

  bool RunOnDevice() override {
    TensorProto::DataType key_dtype = static_cast<TensorProto::DataType>(
        this->template GetSingleArgument<int>(
            "key_dtype", TensorProto_DataType_INT32));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, DataTypeToTypeMeta(key_dtype));
  }

  template <typename KEY_T>
  bool DoRunWithType() {
    TensorProto::DataType value_dtype = static_cast<TensorProto::DataType>(
        this->template GetSingleArgument<int>(
            "value_dtype", TensorProto_DataType_INT32));

    return DispatchHelper<
        TensorTypes2<int32_t, int64_t, GenericTensorImplementation>,
        KEY_T>::call(this, DataTypeToTypeMeta(value_dtype));
  }

  template <typename KEY_T, typename VALUE_T>
  bool DoRunWithType2() {
    // clear to make sure the map is empty
    this->template Output<typename MapTypeTraits<KEY_T, VALUE_T>::MapType>(MAP)
        ->clear();
    return true;
  }

  template <typename KEY_T>
  bool DoRunWithOtherType2() {
    TensorProto::DataType value_dtype = static_cast<TensorProto::DataType>(
        this->template GetSingleArgument<int>(
            "value_dtype", TensorProto_DataType_INT32));

    CAFFE_THROW(
        "CreateMap is not implemented on value tensor of type ",
        DataTypeToTypeMeta(value_dtype).name(),
        "consider adding it as a type in the DispatchHelper list");
  }

  OUTPUT_TAGS(MAP);
};

template <class Context>
class KeyValueToMapOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit KeyValueToMapOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  ~KeyValueToMapOp() override {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(KEYS));
  }

  template <typename KEY_T>
  bool DoRunWithType() {
    return DispatchHelper<
        TensorTypes2<int32_t, int64_t, GenericTensorImplementation>,
        KEY_T>::call(this, Input(VALUES));
  }

  template <typename KEY_T, typename VALUE_T>
  bool DoRunWithType2() {
    using MapType = typename MapTypeTraits<KEY_T, VALUE_T>::MapType;
    const auto& key_input = Input(KEYS);
    const auto& value_input = Input(VALUES);

    CAFFE_ENFORCE_EQ(key_input.numel(), value_input.numel());

    auto* key_data = key_input.template data<KEY_T>();
    auto* value_data = value_input.template data<VALUE_T>();

    auto* map_data = this->template Output<MapType>(MAP);

    for (const auto i : c10::irange(key_input.numel())) {
      map_data->emplace(key_data[i], value_data[i]);
    }

    return true;
  }

  template <typename KEY_T>
  bool DoRunWithOtherType2() {
    CAFFE_THROW(
        "KeyValueToMap is not implemented on value tensor of type ",
        Input(VALUES).dtype().name(),
        "consider adding it as a type in the DispatchHelper list");
  }

  INPUT_TAGS(KEYS, VALUES);
  OUTPUT_TAGS(MAP);
};

template <class Context>
class MapToKeyValueOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit MapToKeyValueOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  ~MapToKeyValueOp() override {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<
        MapType64To64,
        MapType64To32,
        MapType32To32,
        MapType32To64>>::call(this, OperatorBase::InputBlob(MAP));
  }

  template <typename MAP_T>
  bool DoRunWithType() {
    using key_type = typename MAP_T::key_type;
    using mapped_type = typename MAP_T::mapped_type;
    auto& map_data = this->template Input<MAP_T>(MAP);

    auto* key_output = Output(
        KEYS, {static_cast<int64_t>(map_data.size())}, at::dtype<key_type>());
    auto* value_output = Output(
        VALUES,
        {static_cast<int64_t>(map_data.size())},
        at::dtype<mapped_type>());
    auto* key_data = key_output->template mutable_data<key_type>();
    auto* value_data = value_output->template mutable_data<mapped_type>();

    for (const auto& it : map_data) {
      *key_data = it.first;
      *value_data = it.second;
      key_data++;
      value_data++;
    }

    return true;
  }

  INPUT_TAGS(MAP);
  OUTPUT_TAGS(KEYS, VALUES);
};

template <typename KEY_T, typename VALUE_T>
class MapSerializer : public BlobSerializerBase {
 public:
  using MapType = typename MapTypeTraits<KEY_T, VALUE_T>::MapType;

  void Serialize(
      const void* pointer,
      TypeMeta typeMeta,
      const string& name,
      BlobSerializerBase::SerializationAcceptor acceptor) override {
    CAFFE_ENFORCE(typeMeta.Match<MapType>());
    const MapType& map_data = *static_cast<const MapType*>(pointer);
    int64_t sz = map_data.size();
    Tensor key_tensor(CPU);
    key_tensor.Resize(sz);
    Tensor value_tensor(CPU);
    value_tensor.Resize(sz);
    auto* key_data = key_tensor.mutable_data<KEY_T>();
    auto* value_data = value_tensor.mutable_data<VALUE_T>();
    for (const auto& it : map_data) {
      *key_data = it.first;
      *value_data = it.second;
      key_data++;
      value_data++;
    }

    TensorProtos tensor_protos;
    TensorSerializer ser;
    ser.Serialize(
        key_tensor, name, tensor_protos.add_protos(), 0, key_tensor.numel());
    ser.Serialize(
        value_tensor,
        name,
        tensor_protos.add_protos(),
        0,
        value_tensor.numel());

    BlobProto blob_proto;
    blob_proto.set_name(name);
    blob_proto.set_type(MapTypeTraits<KEY_T, VALUE_T>::MapTypeName());
    blob_proto.set_content(SerializeAsString_EnforceCheck(tensor_protos));
    acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
  }
};

template <typename KEY_T, typename VALUE_T>
class MapDeserializer : public BlobDeserializerBase {
 public:
  using MapType = typename MapTypeTraits<KEY_T, VALUE_T>::MapType;

  void Deserialize(const BlobProto& proto, Blob* blob) override {
    TensorProtos tensor_protos;
    CAFFE_ENFORCE(
        tensor_protos.ParseFromString(proto.content()),
        "Fail to parse TensorProtos");
    TensorDeserializer deser;
    Tensor key_tensor = deser.Deserialize(tensor_protos.protos(0));
    Tensor value_tensor = deser.Deserialize(tensor_protos.protos(1));
    auto* key_data = key_tensor.data<KEY_T>();
    auto* value_data = value_tensor.data<VALUE_T>();

    auto* map_ptr = blob->template GetMutable<MapType>();
    for (const auto i : c10::irange(key_tensor.numel())) {
      map_ptr->emplace(key_data[i], value_data[i]);
    }
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MAP_OPS_H_
