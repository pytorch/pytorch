#include "caffe2/core/blob_serialization.h"

#include <sstream>
#include <mutex>

#include "caffe2/core/blob.h"

CAFFE2_DEFINE_int(
    caffe2_tensor_chunk_size,
    1000000,
    "Chunk size to split tensor data into");

namespace caffe2 {

// The blob serialization member function implementation.
void Blob::Serialize(
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor) const {
  std::unique_ptr<BlobSerializerBase> serializer(CreateSerializer(meta_.id()));
  serializer->Serialize(*this, name, acceptor);
}

// The blob serialization member function implementation.
std::string Blob::Serialize(const string& name) const {
  std::stringstream data;
  std::mutex mutex;
  BlobSerializerBase::SerializationAcceptor acceptor =
    [&data, &mutex](const std::string& name, const std::string& blob) {
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
  if (blob_proto.has_tensor()) {
    // This is a tensor object. Depending on the device type, we will
    // use the corresponding TensorDeserializer.
    auto deserializer = CreateDeserializer(
        "Tensor" +
        DeviceType_Name(blob_proto.tensor().device_detail().device_type()));
    // Tensor's deserializer should always be registered, but we will double
    // check if it is not null anyway.
    return CHECK_NOTNULL(deserializer.get())->Deserialize(blob_proto, this);
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
}  // namespace
}  // namespace caffe2
