#include "momentum_sgd_op.h"

namespace caffe2 {

namespace {
REGISTER_CPU_OPERATOR(MomentumSGD, MomentumSGDOp<float, CPUContext>);
OPERATOR_SCHEMA(MomentumSGD)
    .NumInputs(3)
    .NumOutputs(2)
    .AllowInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(

Computes a momentum SGD update for an input gradient and momentum
parameters. Concretely, given inputs (grad, m, lr) and parameters
(momentum, nesterov), computes:

    if not nesterov:
        adjusted_gradient = lr * grad + momentum * m
        return (adjusted_gradient, adjusted_gradient)
    else:
        m_new = momentum * m + lr * grad
        return ((1 + momentum) * m_new - momentum * m, m_new)


)DOC");
SHOULD_NOT_DO_GRADIENT(MomentumSGD);
}
}
