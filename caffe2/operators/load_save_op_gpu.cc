#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/load_save_op.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(LoadTensor, LoadTensorOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Save, SaveOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Snapshot, SnapshotOp<CUDAContext>);
}  // namespace
}  // namespace caffe2
