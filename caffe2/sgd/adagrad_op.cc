#include "adagrad_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Adagrad, AdagradOp<float, CPUContext>);
OPERATOR_SCHEMA(Adagrad)
    .NumInputs(4)
    .NumOutputs(2)
    .AllowInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(

Computes the AdaGrad update for an input gradient and accumulated
history. Concretely, given inputs (param, grad, moment, learning_rate),
computes

    new_moment = moment + square(grad)
    new_grad = learning_rate * grad / (sqrt(new_moment) + epsilon)
    new_param = param + new_grad
and returns (new_param, new_moment).

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment history")
    .Input(2, "grad", "Gradient computed")
    .Input(3, "lr", "learning rate")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment", "Updated moment")
    .Arg("epsilon", "Default 1e-5")
    .Arg(
        "decay",
        "Default 1. If it is in (0, 1), the gradient square sum "
        "is decayed by this factor.");

REGISTER_CPU_OPERATOR(SparseAdagrad, SparseAdagradOp<float, CPUContext>);
OPERATOR_SCHEMA(SparseAdagrad)
    .NumInputs(5)
    .NumOutputs(2)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

Given inputs (param, moment, indices, grad, lr), runs the dense AdaGrad
update on (param, grad, moment[indices], lr), and returns (new_param,
new_moment) as in the dense case.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment history")
    .Input(2, "indices", "Sparse indices")
    .Input(3, "grad", "Gradient computed")
    .Input(4, "lr", "learning rate")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment_1", "Updated moment")
    .Arg("epsilon", "Default 1e-5");

REGISTER_CPU_OPERATOR(
    RowWiseSparseAdagrad,
    RowWiseSparseAdagradOp<float, CPUContext>);
OPERATOR_SCHEMA(RowWiseSparseAdagrad)
    .NumInputs(5)
    .NumOutputs(2)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

Given inputs (param, moment, indices, grad, lr), runs a modified sparse Adagrad
update on (param, grad, moment[indices], lr), and returns (new_param,
new_momwnr), where moment is a 1D tensor with length equal to the number of
rows in param: shape(moment) == shape(param)[0]. Each element of moment is
applied to an entire row of param, and the new moment is calculated by adding
the average squared sum of gradients across each row. Note that indices must
also be a 1D tensor indexing into the rows of param.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment history")
    .Input(2, "indices", "Sparse indices")
    .Input(3, "grad", "Gradient computed")
    .Input(4, "lr", "learning rate")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment_1", "Updated moment")
    .Arg("epsilon", "Default 1e-5");

SHOULD_NOT_DO_GRADIENT(Adagrad);
SHOULD_NOT_DO_GRADIENT(SparseAdagrad);
SHOULD_NOT_DO_GRADIENT(RowWiseSparseAdagrad);
}
