#include "caffe2/sgd/learning_rate_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(LearningRate, LearningRateOp<float, CPUContext>);
}  // namespace
}  // namespace caffe2


