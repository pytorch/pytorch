#include "caffe2/core/context_gpu.h"
#include "caffe2/sgd/iter_op.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(Iter, IterOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(AtomicIter, AtomicIterOp<CUDAContext>);

} // namespace caffe2
