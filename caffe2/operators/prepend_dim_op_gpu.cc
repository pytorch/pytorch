#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/prepend_dim_op.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(PrependDim, PrependDimOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(MergeDim, MergeDimOp<CUDAContext>);

} // namespace caffe2
