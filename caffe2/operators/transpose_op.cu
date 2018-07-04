#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/transpose_op.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(Transpose, TransposeOp<CUDAContext>);

} // namespace caffe2
