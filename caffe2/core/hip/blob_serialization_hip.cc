#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {
REGISTER_BLOB_DESERIALIZER(TensorHIP, TensorDeserializer);
}
} // namespace caffe2
