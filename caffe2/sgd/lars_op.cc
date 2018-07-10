#include "caffe2/sgd/lars_op.h"
#include <math.h>
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
void LarsOp<float, CPUContext>::Compute(
    TIndex N,
    const float* X_data,
    const float* dX_data,
    float offset,
    float* lr_rescale_data) {
  *lr_rescale_data = 1.0;

  float X_norm =
      sqrtf((ConstEigenVectorMap<float>(X_data, N).array()).square().sum());

  if (X_norm > 0) {
    float dX_norm =
        sqrtf((ConstEigenVectorMap<float>(dX_data, N).array()).square().sum());
    *lr_rescale_data /= (dX_norm / X_norm + offset);
  }
}

REGISTER_CPU_OPERATOR(Lars, LarsOp<float, CPUContext>);

OPERATOR_SCHEMA(Lars)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Implement Layer-wise Adaptive Rate Scaling (LARS) as in
https://arxiv.org/abs/1708.03888. Without weight decay, given a global
learning rate lr, parameter tensor X and its gradient dX, the local learning
rate for X will be

    local_lr = lr * norm(X) / ( norm(dX) + offset * norm(X) )

             = lr  / ( norm(dX) / norm(X) + offset ),

where offset is a preset hyper-parameter to avoid numerical issue.
In this implementation, we uses l2 norm and output the rescaling factor

    1 / ( norm(dX) / norm(X) + offset ).

)DOC")
    .Input(0, "X", "Parameter tensor")
    .Input(1, "dX", "Gradient tensor")
    .Output(0, "lr_rescale", "Local learning rate rescaling factor")
    .Arg("offset", "rescaling offset parameter");

SHOULD_NOT_DO_GRADIENT(Lars);
} // namespace caffe2
