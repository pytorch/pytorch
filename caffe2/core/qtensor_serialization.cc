#include "caffe2/core/qtensor_serialization.h"

namespace caffe2 {
namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<QTensor<CPUContext>>()),
    QTensorSerializer<CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_BLOB_DESERIALIZER(QTensor, QTensorDeserializer<CPUContext>);
} // namespace
} // namespace caffe2
