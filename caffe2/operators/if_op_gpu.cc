#include "caffe2/operators/if_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(If, IfOp<CUDAContext>);

} // namespace caffe2
