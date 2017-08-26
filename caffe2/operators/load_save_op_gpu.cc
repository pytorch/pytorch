#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/load_save_op.h"

namespace caffe2 {

template <>
void LoadOp<CUDAContext>::SetCurrentDevice(BlobProto* proto) {
  if (proto->has_tensor()) {
    auto* device_detail = proto->mutable_tensor()->mutable_device_detail();
    device_detail->set_device_type(CUDA);
    device_detail->set_cuda_gpu_id(CaffeCudaGetDevice());
  }
}

REGISTER_CUDA_OPERATOR(Load, LoadOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Save, SaveOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Checkpoint, CheckpointOp<CUDAContext>);
}  // namespace caffe2
