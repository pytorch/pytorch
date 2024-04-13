#include "caffe2/core/context_gpu.h"
#include "caffe2/sgd/learning_rate_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(LearningRate, LearningRateOp<float, CUDAContext>);
}  // namespace caffe2
