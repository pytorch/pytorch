#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/core/typeid.h"
#include "caffe2/core/types.h"

namespace caffe2 {
namespace int8 {

class Int8TensorCPUSerializer : public BlobSerializerBase {
 public:
  void Serialize(
      const Blob& blob,
      const string& name,
      SerializationAcceptor acceptor) override {
    const auto& tensor = blob.template Get<Int8TensorCPU>();
    BlobProto blob_proto;
    blob_proto.set_name(name);
    blob_proto.set_type("Int8TensorCPU");
    QTensorProto& proto = *blob_proto.mutable_qtensor();
    proto.set_name(name);
    for (int i = 0; i < tensor.t.ndim(); ++i) {
      proto.add_dims(tensor.t.dim32(i));
    }
    proto.set_precision(8);
    proto.set_scale(tensor.scale);
    proto.set_bias(tensor.zero_point);
    proto.set_is_signed(false);

    const TensorProto::DataType data_type = TypeMetaToDataType(tensor.t.meta());
    proto.set_data_type(data_type);
    switch (data_type) {
      case TensorProto_DataType_INT32:
        detail::CopyToProtoAsIs(
            tensor.t.size(),
            tensor.t.template data<int32_t>(),
            proto.mutable_data(),
            &this->context_);
        break;
      case TensorProto_DataType_UINT8:
        detail::CopyToProtoWithCast(
            tensor.t.size(),
            tensor.t.template data<uint8_t>(),
            proto.mutable_data(),
            &this->context_);
        break;
      default:
        CAFFE_ENFORCE(false, "Unsupported data type in Int8TensorCPU");
    }

    acceptor(name, blob_proto.SerializeAsString());
  }

 private:
  CPUContext context_;
};

class Int8TensorCPUDeserializer : public TensorDeserializer<CPUContext> {
 public:
  void Deserialize(const BlobProto& blob_proto, Blob* blob) override {
    const QTensorProto& proto = blob_proto.qtensor();
    Int8TensorCPU* tensor = blob->template GetMutable<Int8TensorCPU>();
    tensor->scale = proto.scale();
    tensor->zero_point = proto.bias();
    vector<int> dims;
    for (const int d : proto.dims()) {
      dims.push_back(d);
    }
    tensor->t.Resize(dims);
    switch (proto.data_type()) {
      case TensorProto_DataType_INT32:
        detail::CopyFromProtoAsIs(
            tensor->t.size(),
            proto.data(),
            tensor->t.template mutable_data<int32_t>(),
            &this->context_);
        break;
      case TensorProto_DataType_UINT8:
        detail::CopyFromProtoWithCast(
            tensor->t.size(),
            proto.data(),
            tensor->t.template mutable_data<uint8_t>(),
            &this->context_);
        break;
      default:
        CAFFE_ENFORCE(false, "Unsupported data type in Int8TensorCPU");
    }
  }

 private:
  CPUContext context_;
};

} // namespace int8

namespace {
REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<int8::Int8TensorCPU>()),
    int8::Int8TensorCPUSerializer);
REGISTER_BLOB_DESERIALIZER(Int8TensorCPU, int8::Int8TensorCPUDeserializer);
} // namespace

} // namespace caffe2
