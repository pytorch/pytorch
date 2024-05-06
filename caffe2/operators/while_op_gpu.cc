#include "caffe2/operators/while_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(While, WhileOp<CUDAContext>);

} // namespace caffe2
