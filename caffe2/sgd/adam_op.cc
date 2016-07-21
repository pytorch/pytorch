#include "adam_op.h"

namespace caffe2 {

namespace {
REGISTER_CPU_OPERATOR(Adam, AdamOp<float, CPUContext>);
OPERATOR_SCHEMA(Adam)
    .NumInputs(5)
    .NumOutputs(3)
    .AllowInplace({{0, 0}, {1, 1}, {2, 2}})
    .SetDoc(R"DOC(

Computes the Adam update (https://arxiv.org/abs/1412.6980) for an
input gradient and momentum parameters. Concretely, given inputs
(grad, m1, m2, lr, iters),

    t = iters + 1
    corrected_local_rate = lr * sqrt(1 - power(beta2, t)) /
      (1 - power(beta1, t))
    m1_o = (beta1 * m1) + (1 - beta1) * grad
    m2_o = (beta2 * m2) + (1 - beta2) * np.square(grad)
    grad_o = corrected_local_rate * m1_o / \
        (sqrt(m2_o) + epsilon)

and returns (grad_o, m1_o, m2_o)

)DOC");
SHOULD_NOT_DO_GRADIENT(Adam);
}

}
