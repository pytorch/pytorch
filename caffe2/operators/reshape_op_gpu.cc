#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/reshape_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(Reshape, ReshapeOp<float, CUDAContext>);

} // namespace caffe2
