#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/assert_op.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(Assert, AssertOp<CUDAContext>);

} // namespace caffe2
