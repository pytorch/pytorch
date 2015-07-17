#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"

namespace caffe2 {
DEFINE_TYPED_REGISTRY(BlobSerializerRegistry, internal::TypeId,
                      BlobSerializerBase);

namespace {
REGISTER_BLOB_SERIALIZER(float_cpu,
                         (internal::GetTypeId<Tensor<float, CPUContext> >()),
                         TensorSerializerFloat<CPUContext>);
REGISTER_BLOB_SERIALIZER(int32_cpu,
                         (internal::GetTypeId<Tensor<int, CPUContext> >()),
                         TensorSerializerInt32<CPUContext>);
}
}  // namespace caffe2

