#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/free_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(Free, FreeOp<CUDAContext>);
} // namespace caffe2
