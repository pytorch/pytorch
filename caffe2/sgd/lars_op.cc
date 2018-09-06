#include "caffe2/sgd/lars_op.h"

namespace caffe2 {

template <>
void LarsOp<float, CPUContext>::ComputeLearningRate(
    const float* wd,
    const float* trust,
    const float* lr_max,
    float offset,
    float lr_min,
    float* X_norm,
    float* dX_norm,
    float* lr_rescaled) {
  float val = 1.0;

  if (*X_norm > 0) {
    val = (*trust) / (*dX_norm / *X_norm + (*wd) + offset);
  }
  *lr_rescaled = fmaxf(fminf(val, *lr_max), lr_min);
}

REGISTER_CPU_OPERATOR(Lars, LarsOp<float, CPUContext>);

OPERATOR_SCHEMA(Lars)
    .NumInputs(5)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Implement Layer-wise Adaptive Rate Scaling (LARS) with clipping. Before adding weight
decay, given a parameter tensor X and its gradient dX, the local learning rate
for X will be

local_lr = trust * norm(X) / ( norm(dX) + wd * norm(X) + offset * norm(X) )

      = trust / ( norm(dX) / norm(X) + wd + offset ),

where offset is a preset hyper-parameter to avoid numerical issue and trust
indicates how much we trust the layer to change its parameters during one update.
In this implementation, we uses l2 norm and the computed local learning rate is
clipped based on the upper bound lr_max and the lower bound lr_min:

local_lr = min(local_lr, lr_max) and local_lr = max(local_lr, lr_min)

)DOC")
    .Input(0, "X", "Parameter tensor")
    .Input(1, "dX", "Gradient tensor")
    .Input(2, "wd", "Weight decay")
    .Input(3, "trust", "Trust")
    .Input(4, "lr_max", "Upper bound of learning rate")
    .Output(0, "lr_rescaled", "Rescaled local learning rate")
    .Arg("offset", "rescaling offset parameter")
    .Arg("lr_min", "minimum learning rate for clipping");

SHOULD_NOT_DO_GRADIENT(Lars);
} // namespace caffe2
