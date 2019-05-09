#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {
REGISTER_BLOB_DESERIALIZER(TensorCUDA, TensorDeserializer);
}
}  // namespace caffe2
