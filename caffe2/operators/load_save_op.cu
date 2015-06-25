#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/load_save_op.h"

namespace caffe2 {
namespace {
REGISTER_CUDA_OPERATOR(LoadFloatTensor, LoadFloatTensorOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(SaveFloatTensor, SaveFloatTensorOp<CUDAContext>);
}  // namespace
}  // namespace caffe2
