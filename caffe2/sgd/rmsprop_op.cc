#include "rmsprop_op.h"

namespace caffe2 {

namespace {
REGISTER_CPU_OPERATOR(RmsProp, RmsPropOp<float, CPUContext>);
OPERATOR_SCHEMA(RmsProp)
    .NumInputs(4)
    .NumOutputs(3)
    .AllowInplace({{0, 0}, {1, 1}, {2, 2}})
    .SetDoc(R"DOC(

Computes the RMSProp update
(http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
Concretely, given inputs (grad, mean_squares, mom, lr), computes:

    mean_squares_o = mean_squares + (1 - decay) * (squaare(grad) - mean_squares)
    mom_o = momentum * mom + lr * grad / sqrt(epsilon + mean_squares_o)
    grad_o = mom_o

returns (grad_o, mean_squares_o, mom_o).

)DOC");
SHOULD_NOT_DO_GRADIENT(RmsProp);
}

}
