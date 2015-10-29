#include "caffe2/operators/loss_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(AveragedLoss, AveragedLoss<float, CPUContext>)
REGISTER_CPU_OPERATOR(WeightedSumLoss, WeightedSumLoss<float, CPUContext>)
REGISTER_CPU_OPERATOR(AveragedLossGradient,
                      AveragedLossGradient<float, CPUContext>)
REGISTER_CPU_OPERATOR(WeightedSumLossGradient,
                      WeightedSumLossGradient<float, CPUContext>)

}  // namespace
}  // namespace caffe2
