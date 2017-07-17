#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/accumulate_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(Accumulate, AccumulateOp<float, CUDAContext>);
}  // namespace caffe2
