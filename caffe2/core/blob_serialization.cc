#include "caffe2/core/blob_serialization.h"

#include "caffe2/core/blob.h"

namespace caffe2 {

// The blob serialization member function implementation.
string Blob::Serialize(const string& name) const {
  std::unique_ptr<BlobSerializerBase> serializer(CreateSerializer(meta_.id()));
  return serializer->Serialize(*this, name);
}

// Specialization for StoreDeviceDetail for CPU - nothing needs to be done.
template <>
void TensorSerializer<CPUContext>::StoreDeviceDetail(
    const Tensor<CPUContext>& input, TensorProto* proto) {}

// The actual BlobSerializerRegistry object.
CAFFE_DEFINE_TYPED_REGISTRY(
    BlobSerializerRegistry,
    CaffeTypeId,
    BlobSerializerBase);

namespace {
// Serialize TensorCPU.
REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<TensorCPU>()),
    TensorSerializer<CPUContext>);
}  // namespace
}  // namespace caffe2

