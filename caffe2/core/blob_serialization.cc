#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"

namespace caffe2 {

// The blob serialization member function implementation.
string Blob::Serialize(const string& name) const {
  std::unique_ptr<BlobSerializerBase> serializer(CreateSerializer(meta_.id()));
  return serializer->Serialize(*this, name);
}

// The actual BlobSerializerRegistry object.
DEFINE_TYPED_REGISTRY(BlobSerializerRegistry, CaffeTypeId, BlobSerializerBase);

namespace {
// Serialize TensorCPU.
REGISTER_BLOB_SERIALIZER(tensor_cpu,
                         (TypeMeta::Id<TensorCPU>()),
                         TensorSerializer<CPUContext>);
}  // namespace
}  // namespace caffe2

