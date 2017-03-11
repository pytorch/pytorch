#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/reshape_op.h"

namespace caffe2 {

namespace {

REGISTER_CUDA_OPERATOR(Reshape, ReshapeOp<float, CUDAContext>);

} // namespace
} // namespace caffe2
