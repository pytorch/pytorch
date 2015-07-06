#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {
REGISTER_BLOB_SERIALIZER(float_gpu,
                         (internal::GetTypeId<Tensor<float, CUDAContext> >()),
                         TensorSerializerFloat<CUDAContext>);
REGISTER_BLOB_SERIALIZER(int32_gpu,
                         (internal::GetTypeId<Tensor<int, CUDAContext> >()),
                         TensorSerializerInt32<CUDAContext>);
}
}  // namespace caffe2

