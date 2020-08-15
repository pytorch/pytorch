#include "storm_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Storm, StormOp<CPUContext>);
OPERATOR_SCHEMA(Storm)
    .NumInputs(5)
    .NumOutputs(3)
    .AllowInplace({{0, 0}, {1, 1}, {2, 2}})
    .SetDoc(R"DOC(

Computes the STORM (https://arxiv.org/abs/1905.10018) update for an input
gradient and accumulated history of gradients. Concretely, given inputs
(param, moment, grad_sq_sum, grad, lr), computes:

    new_grad_sq_sum = grad_sq_sum + norm(grad)^2
    effective_lr = lr / (beta + new_grad_sq_sum)^1/3
    alpha = momentum * square(effective_lr)
    new_moment = grad + (1 - alpha) * (moment - grad)
    new_param = param + effective_lr * new_moment

and returns (new_param, new_moment, new_grad_sq_sum).

Note that due to caffe2 limitation, it is difficult to re-calculate gradient
in the previous iteration using the current example. We simplied calculation
for new_moment by using the gradient from the current iteration.

)DOC")
    .Input(0, "param", "Parameters to be updated.")
    .Input(1, "moment", "Moment history.")
    .Input(2, "grad_sq_sum", "Sum of observed squared gradients.")
    .Input(3, "grad", "Gradients computed.")
    .Input(4, "lr", "Learning rate, k in the original paper.")
    .Output(0, "output_param", "Updated parameters.")
    .Output(1, "output_moment", "Updated moment.")
    .Output(2, "output_grad_sq_sum", "Updated sum of squared gradients.")
    .Arg("momentum", "Momentum hyperparameter, c in the original paper.")
    .Arg(
        "beta",
        "denominator in adaptive learning rate, w in the original paper.");

REGISTER_CPU_OPERATOR(SparseStorm, SparseStormOp<CPUContext>);
OPERATOR_SCHEMA(SparseStorm)
    .NumInputs(6)
    .NumOutputs(3)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

This operator implement the STORM (https://arxiv.org/abs/1905.10018)
optimization algorithm. Given inputs (param, moment, grad_sq_sum, grad,
indices, lr), computes the dense STORM update on (param, moment[indices],
grad_sq_sum, grad, lr), and returns (new_param, new_moment, new_grad_sq_sum)
as in the dense case.
)DOC")
    .Input(0, "param", "Parameters to be updated.")
    .Input(1, "moment", "Moment history.")
    .Input(2, "grad_sq_sum", "Sum of observed squared gradients.")
    .Input(3, "grad", "Gradients computed.")
    .Input(4, "indices", "Sparse indices.")
    .Input(5, "lr", "Learning rate, k in the original paper.")
    .Output(0, "output_param", "Updated parameters.")
    .Output(1, "output_moment", "Updated moment.")
    .Output(2, "output_grad_sq_sum", "Updated sum of squared gradients.")
    .Arg("momentum", "Momentum hyperparameter, c in the original paper.")
    .Arg(
        "beta",
        "denominator in adaptive learning rate, w in the original paper.");

SHOULD_NOT_DO_GRADIENT(Storm);
SHOULD_NOT_DO_GRADIENT(SparseStorm);
} // namespace caffe2
