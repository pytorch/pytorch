#include "caffe2/core/blob_serialization.h"

#include <sstream>
#include <mutex>

#include "caffe2/core/blob.h"
#include "caffe2/utils/proto_utils.h"

CAFFE2_DEFINE_int(
    caffe2_tensor_chunk_size,
    1000000,
    "Chunk size to split tensor data into");

CAFFE2_DEFINE_int(
    caffe2_max_tensor_serializer_threads,
    16,
    "Maximal number of threads that can be used for tensor serialization");

CAFFE2_DEFINE_bool(
    caffe2_serialize_fp16_as_bytes,
    false,
    "Serialize FLOAT16 tensors using byte_data field");

namespace caffe2 {
/**
 * @brief StringSerializer is the serializer for String.
 *
 * StringSerializer takes in a blob that contains a String, and serializes it
 * into a BlobProto protocol buffer.
 */
class StringSerializer : public BlobSerializerBase {
 public:
  StringSerializer() {}
  ~StringSerializer() {}
  /**
   * Serializes a Blob. Note that this blob has to contain Tensor<Context>,
   * otherwise this function produces a fatal error.
   */
  void Serialize(
      const Blob& blob,
      const string& name,
      SerializationAcceptor acceptor) override {
    CAFFE_ENFORCE(blob.IsType<std::string>());

    BlobProto blob_proto;
    blob_proto.set_name(name);
    blob_proto.set_type("std::string");
    blob_proto.set_content(blob.template Get<std::string>());
    acceptor(name, blob_proto.SerializeAsString());
  }
};

/**
 * @brief StringDeserializer is the deserializer for Strings.
 *
 */
class StringDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override {
    *blob->GetMutable<std::string>() = proto.content();
  }
};

// The blob serialization member function implementation.
void Blob::Serialize(
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor,
    int chunk_size) const {
  std::unique_ptr<BlobSerializerBase> serializer(CreateSerializer(meta_.id()));
  CAFFE_ENFORCE(serializer, "No known serializer for ", meta_.name());
  serializer->SerializeWithChunkSize(*this, name, acceptor, chunk_size);
}

// The blob serialization member function implementation.
std::string Blob::Serialize(const string& name) const {
  std::string data;
  BlobSerializerBase::SerializationAcceptor acceptor = [&data](
      const std::string&, const std::string& blob) {
    DCHECK(data.empty()); // should be called once with kNoChunking
    data = blob;
  };
  this->Serialize(name, acceptor, kNoChunking);
  return data;
}

// Specialization for StoreDeviceDetail for CPU - nothing needs to be done.
template <>
void TensorSerializer<CPUContext>::StoreDeviceDetail(
    const Tensor<CPUContext>& /*input*/,
    TensorProto* /*proto*/) {}

// The actual serialization registry objects.
CAFFE_DEFINE_TYPED_REGISTRY(
    BlobSerializerRegistry,
    CaffeTypeId,
    BlobSerializerBase,
    std::unique_ptr);

CAFFE_DEFINE_REGISTRY(BlobDeserializerRegistry, BlobDeserializerBase);

void Blob::Deserialize(const string& content) {
  BlobProto blob_proto;
  CAFFE_ENFORCE(
      blob_proto.ParseFromString(content),
      "Cannot parse content into a BlobProto.");
  Deserialize(blob_proto);
}

void Blob::Deserialize(const BlobProto& blob_proto) {
  if (blob_proto.type() == kTensorBlobType) {
    // This is a tensor object. Depending on the device type, we will
    // use the corresponding TensorDeserializer.
    auto deserializer = CreateDeserializer(
        "Tensor" +
        DeviceTypeName(blob_proto.tensor().device_detail().device_type()));
    // Tensor's deserializer should always be registered, but we will double
    // check if it is not null anyway.
    CAFFE_ENFORCE(deserializer.get());
    deserializer->Deserialize(blob_proto, this);
  } else {
    auto deserializer = CreateDeserializer(blob_proto.type());
    CAFFE_ENFORCE(
        deserializer.get(),
        "No registered deserializer for type ",
        blob_proto.type());
    deserializer->Deserialize(blob_proto, this);
  }
}

namespace {
// Serialize TensorCPU.
REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<TensorCPU>()),
    TensorSerializer<CPUContext>);
REGISTER_BLOB_DESERIALIZER(TensorCPU, TensorDeserializer<CPUContext>);
// Serialize std::string
REGISTER_BLOB_SERIALIZER((TypeMeta::Id<std::string>()), StringSerializer);
REGISTER_BLOB_DESERIALIZER(std::string, StringDeserializer);
}  // namespace
}  // namespace caffe2
