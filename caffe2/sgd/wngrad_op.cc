#include "wngrad_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Wngrad, WngradOp<float, CPUContext>);
OPERATOR_SCHEMA(Wngrad)
    .NumInputs(4)
    .NumOutputs(2, 4)
    .AllowInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(

Computes the WnGrad update for an input gradient and accumulated
history. This operator implement the optimization algorithm
in https://arxiv.org/abs/1803.02865 by Wu, Ward and Bottou.
Concretely, given inputs (param, grad, seq_b, learning_rate),
computes

    new_seq_b = seq_b + 1 / seq_b * norm(grad)^2
    effective_lr = learning_rate / (new_seq_b + epsilon)
    update = learning_rate * grad / (new_seq_b + epsilon)
    new_param = param + update
and returns (new_param, new_seq_b).

Optionally returns effective_lr and update as well.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "seq_b", "Seq_b history")
    .Input(2, "grad", "Gradient computed")
    .Input(3, "lr", "learning rate")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_seq_b", "Updated seq_b")
    .Output(2, "output_effective_lr", "(optional) Effective learning rate")
    .Output(3, "output_update", "(optional) Actual update that is applied.")

    .Arg("epsilon", "Default 1e-5");

REGISTER_CPU_OPERATOR(SparseWngrad, SparseWngradOp<float, CPUContext>);
OPERATOR_SCHEMA(SparseWngrad)
    .NumInputs(5)
    .NumOutputs(2)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

This operator implement the optimization algorithm
in https://arxiv.org/abs/1803.02865 by Wu, Ward and Bottou.
Given inputs (param, seq_b, indices, grad, lr), runs the dense WnGrad
update on (param, grad, seq_b, lr), and returns (new_param,
new_seq_b) as in the dense case.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "seq_b", "seq_b history")
    .Input(2, "indices", "Sparse indices")
    .Input(3, "grad", "Gradient computed")
    .Input(4, "lr", "learning rate")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_seq_b", "Updated seq_b")
    .Arg("epsilon", "Default 1e-5");

SHOULD_NOT_DO_GRADIENT(Wngrad);
SHOULD_NOT_DO_GRADIENT(SparseWngrad);
} // namespace caffe2
