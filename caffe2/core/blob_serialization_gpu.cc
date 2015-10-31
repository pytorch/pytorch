#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {
namespace {
REGISTER_BLOB_SERIALIZER(tensor_gpu,
                         (TypeMeta::Id<TensorCUDA>()),
                         TensorSerializer<CUDAContext>);
}
}  // namespace caffe2

