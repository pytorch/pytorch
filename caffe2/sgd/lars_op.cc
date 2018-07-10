#include "caffe2/sgd/lars_op.h"
#include <math.h>
#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
void LarsOp<float, CPUContext>::Compute(
    TIndex N,
    const float* dX_data,
    const float* momentum,
    float offset,
    float* lr_rescale_data) {
  *lr_rescale_data = 1.0;

  float dX_norm =
      sqrtf((ConstEigenVectorMap<float>(dX_data, N).array()).square().sum());
  float momentum_norm =
      sqrtf((ConstEigenVectorMap<float>(momentum, N).array()).square().sum());

  float numer = ConstEigenVectorMap<float>(dX_data, N)
                    .dot(ConstEigenVectorMap<float>(momentum, N));

  float scale = numer / (dX_norm * momentum_norm + offset);

  *lr_rescale_data = scale;
}

REGISTER_CPU_OPERATOR(Lars, LarsOp<float, CPUContext>);

OPERATOR_SCHEMA(Lars)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Implement Layer-wise Adaptive Rate Scaling (LARS) as in
https://arxiv.org/abs/1708.03888. After weight decay, given a global
learning rate lr, parameter gradient dX and its momentum, the local learning
rate for parameter X will be

    local_lr = lr * (dX).dot(momentum) / ( norm(dX)*norm(momentum) + offset),

where offset is a preset hyper-parameter to avoid numerical issue.
In this implementation, we uses l2 norm and output the rescaling factor

    (dX).dot(momentum) / ( norm(dX)*norm(momentum) + offset).

)DOC")
    .Input(0, "dX", "Gradient tensor")
    .Input(1, "momentum", "momentum tensor")
    .Output(0, "lr_rescale", "Local learning rate rescaling factor")
    .Arg("offset", "rescaling offset parameter");

SHOULD_NOT_DO_GRADIENT(Lars);
} // namespace caffe2
