#include "adagrad_op.h"

namespace caffe2 {

namespace {
REGISTER_CPU_OPERATOR(Adagrad, AdagradOp<float, CPUContext>);
OPERATOR_SCHEMA(Adagrad)
    .NumInputs(3)
    .NumOutputs(2)
    .AllowInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(

Computes the AdaGrad update for an input gradient and accumulated
history. Concretely, given inputs (grad, history, learning_rate), computes

    new_history = history + square(grad)
    new_grad = learning_rate * grad / (sqrt(new_history) + epsilon)

and returns (new_grad, new_history).

)DOC");

REGISTER_CPU_OPERATOR(SparseAdagrad, SparseAdagradOp<float, CPUContext>);
OPERATOR_SCHEMA(SparseAdagrad)
    .NumInputs(4)
    .NumOutputs(2)
    .AllowInplace({{1, 0}, {2, 1}})
    .SetDoc(R"DOC(

Given inputs (indices, grad, history, lr), runs the dense AdaGrad
update on (grad, history[indices], lr), and returns (new_grad,
new_history) as in the dense case.

)DOC");
SHOULD_NOT_DO_GRADIENT(Adagrad);
SHOULD_NOT_DO_GRADIENT(SparseAdagrad);
}
}
