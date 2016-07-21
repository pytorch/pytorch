#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/scale_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(Scale, ScaleOp<float, CUDAContext>);
}  // namespace caffe2
