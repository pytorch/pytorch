#include "caffe2/sgd/learning_rate_adaption_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    LearningRateAdaption,
    LearningRateAdaptionOp<float, CPUContext>);

OPERATOR_SCHEMA(LearningRateAdaption)
    .NumInputs(3)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
      Learning Rate Adaption is an operation that perform one iteration of
      gradient descent based on learning rate:
        lr(k) = lr(k-1) - lr_alpha * df(k-1)/dlr,
      where df(k-1)/dlr is the gradient of objective function f on lr, and
      lr_alpha is a learning rate hyperparameter. It can be prove that
      df(k-1)/dlr equals INNERPRODUCT(grad(k-1), -grad(k-2)), where grad(k-1) is
      the grad of f(k-1) on parameters. When the argument
      "normalized_lr_adaption" is false, we simply perform the
      following update:
      lr(k) = lr(k-1) - lr_alpha * INNERPRODUCT(grad(k-1), grad(k-2)).
      If we set "normalized_lr_adaption" to be true, we do not directly apply
      INNERPRODUCT(grad(k-1), -grad(k-2)) as the grad. Instead, we perform the
      following update:
      lr(k) = lr(k-1) + lr_alpha * cosineSimilarity(grad(k-1), grad(k-2)).
)DOC")
    .Arg(
        "lr_alpha",
        "the learning rate for performing gradient descent on learning rate lr")
    .Arg(
        "normalized_lr_adaption",
        "whether to apply normalized lr adaption or not")
    .Input(0, "lr", "Learning rate")
    .Input(1, "grad", "Gradient computed")
    .Input(2, "effgrad", "The effective grad")
    .Output(0, "output_lr", "Updated learning rate");

NO_GRADIENT(LearningRateAdaption);
} // namespace caffe2
