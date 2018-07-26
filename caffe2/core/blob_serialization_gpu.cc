#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

template <>
void TensorSerializer<CUDAContext>::StoreDeviceDetail(
    const Tensor<CUDAContext>& input, TensorProto* proto) {
  auto* device_detail = proto->mutable_device_detail();
  device_detail->set_device_type(CUDA);
  device_detail->set_cuda_gpu_id(
      GetGPUIDForPointer(input.raw_data()));
}

namespace {
REGISTER_BLOB_SERIALIZER(
    (TypeMeta::Id<TensorCUDA>()),
    TensorSerializer<CUDAContext>);
REGISTER_BLOB_DESERIALIZER(TensorCUDA, TensorDeserializer<CUDAContext>);
}
}  // namespace caffe2

