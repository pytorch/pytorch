#include "caffe2/operators/order_switch_ops.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(NHWC2NCHW, NHWC2NCHWOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(NCHW2NHWC, NCHW2NHWCOp<float, CUDAContext>);

} // namespace caffe2
