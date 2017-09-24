#include "caffe2/sgd/learning_rate_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(LearningRate, LearningRateOp<float, CPUContext>);

OPERATOR_SCHEMA(LearningRate)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Learning rate is a decreasing function of time. With low learning rates the
improvements will be linear. With high learning rates they will start to look
more exponential. Learning rate is controlled by the following arguments:

#### Required

* `iterations`
* `base_lr`: base learning rate
* `policy`: this controls how the learning rate is applied, options are:
  * `fixed`
  * `step`: uses `stepsize`, `gamma`
  * `exp`: uses `gamma`
  * `inv`: uses `gamma`, `power`

#### Optional:
* `stepsize`: defaults to 0
* `gamma`: defaults to 0
* `power`: defaults to 0

Usage:
train_net.LearningRate(*iterations*, "*label*", base_lr=*float*,
                            policy="policy_name", stepsize=*int*, gamma=*float*)
Example usage:
train_net.LearningRate(200, "LR", base_lr=-0.1,
                            policy="step", stepsize=20, gamma=0.9)
)DOC")
    .Arg("base_lr", "(float, required) base learning rate")
    .Arg("policy", "(float, default 1.0) strategy for gamma enforcement")
    .Arg("power", "(float, default 1.0) used only for inv policy type")
    .Arg("gamma", "(float, default 1.0) momentum of change")
    .Arg("stepsize", "(float, default 1.0) sampling rate on iterations")
    .Arg(
        "max_iter",
        "(int, default -1) maximum iterations in this training run")
    .Input(0, "input", "description needed")
    .Output(0, "output", "description needed")
    .DeviceInferenceFunction([](const OperatorDef& def) {
      return std::make_pair(
          std::vector<DeviceOption>{DeviceOption()},
          std::vector<DeviceOption>{def.device_option()});
    });

NO_GRADIENT(LearningRate);
}  // namespace caffe2
