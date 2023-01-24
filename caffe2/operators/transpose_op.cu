#include "caffe2/operators/transpose_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(Transpose, TransposeOp<CUDAContext>);

} // namespace caffe2
