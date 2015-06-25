#include "caffe2/operators/loss_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(AveragedLoss, AveragedLoss<float, CPUContext>)
REGISTER_CPU_OPERATOR(WeightedSumLoss, WeightedSumLoss<float, CPUContext>)

}  // namespace
}  // namespace caffe2
