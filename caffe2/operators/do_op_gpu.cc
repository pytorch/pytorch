#include "caffe2/operators/do_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(Do, DoOp<CUDAContext>);

} // namespace caffe2
