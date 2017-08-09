#include "caffe2/operators/map_ops.h"

namespace caffe2 {

using MapType64To64 = MapTypeTraits<int64_t, int64_t>::MapType;
CAFFE_KNOWN_TYPE(MapType64To64);

using MapType64To32 = MapTypeTraits<int64_t, int32_t>::MapType;
CAFFE_KNOWN_TYPE(MapType64To32);

using MapType32To32 = MapTypeTraits<int32_t, int32_t>::MapType;
CAFFE_KNOWN_TYPE(MapType32To32);

using MapType32To64 = MapTypeTraits<int32_t, int64_t>::MapType;
CAFFE_KNOWN_TYPE(MapType32To64);

namespace {

REGISTER_BLOB_SERIALIZER(
    TypeMeta::Id<MapType64To64>(),
    MapSerializer<int64_t, int64_t>);

REGISTER_BLOB_SERIALIZER(
    TypeMeta::Id<MapType64To32>(),
    MapSerializer<int64_t, int32_t>);

REGISTER_BLOB_SERIALIZER(
    TypeMeta::Id<MapType32To32>(),
    MapSerializer<int32_t, int32_t>);

REGISTER_BLOB_SERIALIZER(
    TypeMeta::Id<MapType32To64>(),
    MapSerializer<int32_t, int64_t>);

REGISTER_BLOB_DESERIALIZER(
    (std::unordered_map<int64_t, int64_t>),
    MapDeserializer<int64_t, int64_t>);

REGISTER_BLOB_DESERIALIZER(
    (std::unordered_map<int64_t, int32_t>),
    MapDeserializer<int64_t, int32_t>);

REGISTER_BLOB_DESERIALIZER(
    (std::unordered_map<int32_t, int32_t>),
    MapDeserializer<int32_t, int32_t>);

REGISTER_BLOB_DESERIALIZER(
    (std::unordered_map<int32_t, int64_t>),
    MapDeserializer<int32_t, int64_t>);

REGISTER_CPU_OPERATOR(CreateMap, CreateMapOp<CPUContext>);
REGISTER_CPU_OPERATOR(KeyValueToMap, KeyValueToMapOp<CPUContext>);
REGISTER_CPU_OPERATOR(MapToKeyValue, MapToKeyValueOp<CPUContext>);

OPERATOR_SCHEMA(CreateMap)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc("Create an empty map blob")
    .Arg("key_dtype", "Key's TensorProto::DataType (default INT32)")
    .Arg("value_dtype", "Value's TensorProto::DataType (default INT32)")
    .Output(0, "map blob", "Blob reference to the map");

OPERATOR_SCHEMA(KeyValueToMap)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc("Convert key and value blob pairs into a map blob")
    .Input(0, "key blob", "Blob reference to the key")
    .Input(1, "value blob", "Blob reference to the value")
    .Output(0, "map blob", "Blob reference to the map");

OPERATOR_SCHEMA(MapToKeyValue)
    .NumInputs(1)
    .NumOutputs(2)
    .SetDoc("Convert a map blob into key and value blob pairs")
    .Input(0, "map blob", "Blob reference to the map")
    .Output(0, "key blob", "Blob reference to the key")
    .Output(1, "value blob", "Blob reference to the value");
}
} // namespace caffe2
