#include "caffe2/core/blob_serialization.h"

#include <sstream>
#include <mutex>

#include "caffe2/core/blob.h"

CAFFE2_DEFINE_int(
    caffe2_tensor_chunk_size,
    1000000,
    "Chunk size to split tensor data into");

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
  bool Deserialize(const BlobProto& proto, Blob* blob) override {
    *blob->GetMutable<std::string>() = proto.content();
    return true;
  }
};

namespace {

// A wrapper function to return tensor type string appended with the device
// name, for use in blob serialization / deserialization. This should have one
// to one correspondence with caffe2/proto/caffe2.proto: enum DeviceType.
//
// Note that we can't use DeviceType_Name, because that is only available in
// protobuf-full, and some platforms (like mobile) may want to use
// protobuf-lite instead.
std::string TensorDeviceTypeName(const int32_t& d) {
  switch (d) {
    case CPU:
      return "TensorCPU";
    case CUDA:
      return "TensorCUDA";
    case MKLDNN:
      return "TensorMKLDNN";
    default:
      CAFFE_THROW("Unknown device: ", d);
      return "";
  }
};
}

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
  std::stringstream data;
  std::mutex mutex;
  BlobSerializerBase::SerializationAcceptor acceptor =
    [&data, &mutex](const std::string&, const std::string& blob) {
    std::lock_guard<std::mutex> guard(mutex);
    data << blob;
  };
  this->Serialize(name, acceptor);
  return data.str();
}

// Specialization for StoreDeviceDetail for CPU - nothing needs to be done.
template <>
void TensorSerializer<CPUContext>::StoreDeviceDetail(
    const Tensor<CPUContext>& input, TensorProto* proto) {}

// The actual serialization registry objects.
CAFFE_DEFINE_TYPED_REGISTRY(
    BlobSerializerRegistry,
    CaffeTypeId,
    BlobSerializerBase);

CAFFE_DEFINE_REGISTRY(BlobDeserializerRegistry, BlobDeserializerBase);

bool Blob::Deserialize(const string& content) {
  BlobProto blob_proto;
  if (!blob_proto.ParseFromString(content)) {
    LOG(ERROR) << "Cannot parse content into a BlobProto.";
    return false;
  }
  return Deserialize(blob_proto);
}

bool Blob::Deserialize(const BlobProto& blob_proto) {
  if (blob_proto.type() == kTensorBlobType) {
    // This is a tensor object. Depending on the device type, we will
    // use the corresponding TensorDeserializer.
    auto deserializer = CreateDeserializer(TensorDeviceTypeName(
        blob_proto.tensor().device_detail().device_type()));
    // Tensor's deserializer should always be registered, but we will double
    // check if it is not null anyway.
    CAFFE_ENFORCE(deserializer.get());
    return deserializer->Deserialize(blob_proto, this);
  } else {
    auto deserializer = CreateDeserializer(blob_proto.type());
    if (!deserializer.get()) {
      LOG(ERROR) << "No registered deserializer for type " << blob_proto.type();
      return false;
    }
    return deserializer->Deserialize(blob_proto, this);
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
