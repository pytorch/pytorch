#include "momentum_sgd_op.h"

namespace caffe2 {

namespace {
REGISTER_CPU_OPERATOR(MomentumSGD, MomentumSGDOp<float, CPUContext>);
OPERATOR_SCHEMA(MomentumSGD)
    .NumInputs(3)
    .NumOutputs(2)
    .AllowInplace({{0, 0}, {1, 1}})
    .TensorInferenceFunction(
        [](const OperatorDef& /* unused */, const vector<TensorShape>& in) {
          vector<TensorShape> out(2);
          out[0] = in[0];
          out[1] = in[1];
          return out;
        })
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

Output is (grad, momentum)

Note the difference to MomemtumSGDUpdate, which actually performs the
parameter update (and is thus faster).
)DOC");
SHOULD_NOT_DO_GRADIENT(MomentumSGD);

REGISTER_CPU_OPERATOR(
    MomentumSGDUpdate,
    MomentumSGDUpdateOp<float, CPUContext>);
OPERATOR_SCHEMA(MomentumSGDUpdate)
    .NumInputs(4)
    .NumOutputs(3)
    .AllowInplace({{0, 0}, {1, 1}, {3, 2}})
    .TensorInferenceFunction(
        [](const OperatorDef& /* unused */, const vector<TensorShape>& in) {
          vector<TensorShape> out(3);
          out[0] = in[0];
          out[1] = in[1];
          out[2] = in[3];
          return out;
        })
    .SetDoc(R"DOC(

Performs a momentum SGD update for an input gradient and momentum
parameters. Concretely, given inputs (grad, m, lr, param) and parameters
(momentum, nesterov), computes:

    if not nesterov:
        adjusted_gradient = lr * grad + momentum * m
        param = param - adjusted_gradient
        return (adjusted_gradient, adjusted_gradient, param)
    else:
        m_new = momentum * m + lr * grad
        param = param - ((1 + momentum) * m_new - momentum * m),
        return ((1 + momentum) * m_new - momentum * m, m_new, param)

Output is (grad, momentum, parameter).

Note the difference to MomentumSGD, which returns a new gradient
but does not perform the parameter update.

)DOC");
SHOULD_NOT_DO_GRADIENT(MomentumSGDUpdate);
}
}
