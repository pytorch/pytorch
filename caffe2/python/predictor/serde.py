## @package serde
# Module caffe2.python.predictor.serde






def serialize_protobuf_struct(protobuf_struct):
    return protobuf_struct.SerializeToString()


def deserialize_protobuf_struct(serialized_protobuf, struct_type):
    deser = struct_type()
    deser.ParseFromString(serialized_protobuf)
    return deser
