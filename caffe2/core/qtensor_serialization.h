#ifndef CAFFE2_CORE_QTENSOR_SERIALIZATION_H_
#define CAFFE2_CORE_QTENSOR_SERIALIZATION_H_

#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/qtensor.h"

namespace caffe2 {

constexpr auto kQTensorBlobQType = "QTensor";

template <class Context>
class QTensorSerializer : public BlobSerializerBase {
 public:
  QTensorSerializer() : context_() {}
  ~QTensorSerializer() {}
  /**
   * Serializes a Blob. Note that this blob has to contain QTensor<Context>.
   */
  void Serialize(
      const void* pointer,
      TypeMeta typeMeta,
      const string& name,
      SerializationAcceptor acceptor) override;

 private:
  Context context_;
};

template <class Context>
class QTensorDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override;
  void Deserialize(const QTensorProto& proto, QTensor<Context>* tensor);
};

template <class Context>
void QTensorSerializer<Context>::Serialize(
    const void* pointer,
    TypeMeta typeMeta,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor) {
  CAFFE_ENFORCE(typeMeta.Match<QTensor<Context>>());
  const auto& qtensor = *static_cast<const QTensor<Context>*>(pointer);
  BlobProto blob_proto;
  blob_proto.set_name(name);
  blob_proto.set_type(kQTensorBlobQType);
  QTensorProto& proto = *blob_proto.mutable_qtensor();
  proto.set_name(name);
  for (int i = 0; i < qtensor.ndim(); ++i) {
    proto.add_dims(qtensor.dim32(i));
  }
  proto.set_precision(qtensor.precision());
  proto.set_scale(qtensor.scale());
  proto.set_bias(qtensor.bias());
  proto.set_is_signed(qtensor.is_signed());
  detail::CopyToProtoWithCast(
      qtensor.nbytes(), qtensor.data(), proto.mutable_data(), &this->context_);
  acceptor(name, SerializeBlobProtoAsString_EnforceCheck(blob_proto));
}

template <class Context>
void QTensorDeserializer<Context>::Deserialize(
    const BlobProto& blob_proto,
    Blob* blob) {
  Deserialize(blob_proto.qtensor(), blob->GetMutable<QTensor<Context>>());
}

template <class Context>
void QTensorDeserializer<Context>::Deserialize(
    const QTensorProto& proto,
    QTensor<Context>* qtensor) {
  Context context{};
  vector<int> dims;
  for (const int d : proto.dims()) {
    dims.push_back(d);
  }
  qtensor->Resize(dims);
  qtensor->SetPrecision(proto.precision());
  qtensor->SetScale(proto.scale());
  qtensor->SetBias(proto.bias());
  qtensor->SetSigned(proto.is_signed());

  detail::CopyFromProtoWithCast(
      qtensor->nbytes(), proto.data(), qtensor->mutable_data(), &context);
}

} // namespace caffe2

#endif // CAFFE2_CORE_QTENSOR_SERIALIZATION_H_
