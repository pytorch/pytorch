#include "caffe2/sgd/adadelta_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Adadelta, AdadeltaOp<CPUContext>);
OPERATOR_SCHEMA(Adadelta)
    .NumInputs(5)
    .NumOutputs(3)
    .AllowInplace({{0, 0}, {1, 1}, {2, 2}})
    .SetDoc(R"DOC(

Computes the AdaDelta update (https://arxiv.org/abs/1212.5701) for an input
gradient and accumulated history of squared gradients. Concretely, given
inputs (param, moment, moment_delta, grad, learning_rate), computes:

    new_moment = moment * decay + square(grad) * (1 - decay)
    new_grad = sqrt(moment_delta + epsilon) / sqrt(new_moment + epsilon) * grad
    new_param = param + learning_rate * new_grad
    new_moment_delta = moment_delta * decay + square(new_grad) * (1 - decay)

and returns (new_param, new_moment, new_moment_delta).

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Average of squared gradients")
    .Input(2, "moment_delta", "Average of squared parameter updates")
    .Input(3, "grad", "Gradient computed")
    .Input(4, "lr", "Learning rate")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment", "Updated average squared gradient")
    .Output(
        2,
        "output_moment_delta",
        "Updated average of squared parameter updates")
    .Arg("epsilon", "Default 1e-5")
    .Arg(
        "decay",
        "Default 0.95, the squared gradient sum is decayed by this factor.");

REGISTER_CPU_OPERATOR(SparseAdadelta, SparseAdadeltaOp<CPUContext>);
OPERATOR_SCHEMA(SparseAdadelta)
    .NumInputs(6)
    .NumOutputs(3)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

Given inputs (param, moment, moment_delta, indices, grad, lr),
runs the dense AdaDelta update on (param, grad, moment[indices],
 moment_delta[indices], lr), and returns (new_param, new_moment,
 new_moment_delta) as in the dense case.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Average of squared gradients")
    .Input(2, "moment_delta", "Average of squared parameter updates")
    .Input(3, "indices", "Sparse indices")
    .Input(4, "grad", "Gradient computed")
    .Input(5, "lr", "learning rate")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment", "Updated average squared gradient")
    .Output(
        2,
        "output_moment_delta",
        "Updated average of squared parameter updates")
    .Arg("epsilon", "Default 1e-5")
    .Arg(
        "decay",
        "Default 0.95, the squared gradient sum is decayed by this factor.");

SHOULD_NOT_DO_GRADIENT(Adadelta);
SHOULD_NOT_DO_GRADIENT(SparseAdadelta);
} // namespace caffe2
