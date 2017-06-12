#ifndef CAFFE2_OPERATORS_MAP_OPS_H_
#define CAFFE2_OPERATORS_MAP_OPS_H_

#include <algorithm>
#include <iterator>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

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
class KeyValueToMapOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  KeyValueToMapOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~KeyValueToMapOp() {}

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

    CAFFE_ENFORCE_EQ(key_input.size(), value_input.size());

    auto* key_data = key_input.template data<KEY_T>();
    auto* value_data = value_input.template data<VALUE_T>();

    auto* map_data = OperatorBase::Output<MapType>(MAP);

    for (int i = 0; i < key_input.size(); ++i) {
      map_data->emplace(key_data[i], value_data[i]);
    }

    return true;
  }

  template <typename KEY_T>
  bool DoRunWithOtherType2() {
    CAFFE_THROW(
        "KeyValueToMap is not implemented on value tensor of type ",
        Input(VALUES).meta().name(),
        "Consider adding it a type in the list DispatchHelper");
  }

  INPUT_TAGS(KEYS, VALUES);
  OUTPUT_TAGS(MAP);
};

template <class Context>
class MapToKeyValueOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MapToKeyValueOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  ~MapToKeyValueOp() {}

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
    auto& map_data = OperatorBase::Input<MAP_T>(MAP);
    auto* key_output = Output(KEYS);
    auto* value_output = Output(VALUES);
    key_output->Resize(map_data.size());
    value_output->Resize(map_data.size());
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
      const Blob& blob,
      const string& name,
      BlobSerializerBase::SerializationAcceptor acceptor) override {
    CAFFE_ENFORCE(blob.IsType<MapType>());
    const MapType& map_data = blob.template Get<MapType>();
    TIndex sz = map_data.size();
    Tensor<CPUContext> key_tensor;
    key_tensor.Resize(sz);
    Tensor<CPUContext> value_tensor;
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
    TensorSerializer<CPUContext> ser;
    ser.Serialize(
        key_tensor, name, tensor_protos.add_protos(), 0, key_tensor.size());
    ser.Serialize(
        value_tensor, name, tensor_protos.add_protos(), 0, value_tensor.size());

    BlobProto blob_proto;
    blob_proto.set_name(name);
    blob_proto.set_type(MapTypeTraits<KEY_T, VALUE_T>::MapTypeName());
    blob_proto.set_content(tensor_protos.SerializeAsString());
    acceptor(name, blob_proto.SerializeAsString());
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
    TensorDeserializer<CPUContext> deser;
    Tensor<CPUContext> key_tensor, value_tensor;
    deser.Deserialize(tensor_protos.protos(0), &key_tensor);
    deser.Deserialize(tensor_protos.protos(1), &value_tensor);
    auto* key_data = key_tensor.data<KEY_T>();
    auto* value_data = value_tensor.data<VALUE_T>();

    auto* map_ptr = blob->template GetMutable<MapType>();
    for (int i = 0; i < key_tensor.size(); ++i) {
      map_ptr->emplace(key_data[i], value_data[i]);
    }
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MAP_OPS_H_
