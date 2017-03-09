#include "caffe2/core/qtensor_serialization.h"

namespace caffe2 {
namespace {
REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<QTensor<CPUContext>>()),
    QTensorSerializer<CPUContext>);
REGISTER_BLOB_DESERIALIZER(QTensor, QTensorDeserializer<CPUContext>);
} // namespace
} // namespace caffe2
