#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

template <>
void TensorSerializer<HIPContext>::StoreDeviceDetail(const Tensor<HIPContext>& input,
                                                     TensorProto* proto)
{
    auto* device_detail = proto->mutable_device_detail();
    device_detail->set_device_type(HIP);
    device_detail->set_hip_gpu_id(GetGPUIDForPointer(input.raw_data()));
}

namespace {
REGISTER_BLOB_SERIALIZER((TypeMeta::Id<TensorHIP>()), TensorSerializer<HIPContext>);
REGISTER_BLOB_DESERIALIZER(TensorHIP, TensorDeserializer<HIPContext>);
}
} // namespace caffe2
