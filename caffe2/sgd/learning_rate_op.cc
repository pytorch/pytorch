#include "caffe2/sgd/learning_rate_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(LearningRate, LearningRateOp<float, CPUContext>);

OPERATOR_SCHEMA(LearningRate)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef&,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0] = in[0];
      return out;
    })
    .SetDoc(R"DOC(
Learning rate is a decreasing function of time. With low learning rates the
improvements will be linear. With high learning rates they will start to look
more exponential. Learning rate is controlled by the following arguments:


Required:
 `iterations`
 `base_lr`: base learning rate
 `policy`: this controls how the learning rate is applied, options are:
   `fixed`
   `step`: uses `stepsize`, `gamma`
   `exp`: uses `gamma`
   `gate`: uses 'multiplier_1', 'multiplier_2', `num_iter``
   `inv`: uses `gamma`, `power`
   `linearWarmup`: uses `start_multiplier`, `num_iter`
   `constantWarmup`: uses `multiplier`, `num_iter`
   `alter`: uses  `active_first`, `active_period`, `inactive_period`
   `hill`: uses those in both `linearWarmup` and `inv`, plus `end_multiplier`
   `composite`: uses `sub_policy_num_iters` and additional args with format
   `cyclic`: uses `max_lr`, `stepsize`
   sub_policy_{sub_policy_index}_{sub_policy_arg}, for example:
   sub_policy_0_policy: "exp", sub_policy_0_gamma: 0.99,
   sub_policy_0_lr_scale: 1.2
   sub_policy_0_policy: "fixed", sub_policy_0_lr_scale: 1.0
   sub_policy_num_iters: [1000, 1000]

Optional:
  `stepsize`: defaults to 0
  `max_lr`: defaults to 0.005
  `gamma`: defaults to 0
  `power`: defaults to 0
  `num_iter`: defaults to 0
  `start_multiplier`: defaults to 0
  `multiplier`: defaults to 0.5
  `multiplier_1`: defaults to 1
  `multiplier_2`: defaults to 1
  `m1`: defaults to 0.5, the first piece lr of piece warmup
  `n1`: defaults to 0, iter threshold of the first piece lr
  `m2`: defaults to 0.5, the second piece lr of piece warmup
  `n2`: defaults to 0, iter threshold of the second piece lr
  `m3`: defaults to 0.5, the third piece lr of piece warmup


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
    .Arg("max_lr", "(float, default 0.005) max learning rate")
    .Arg("active_first", "(boolean, default True) in alter policy")
    .Arg("active_period", "(int64_t, required) in alter policy")
    .Arg("inactive_period", "(int64_t, required) in alter policy")
    .Arg(
        "max_iter",
        "(int, default -1) maximum iterations in this training run")
    .Arg(
        "num_iter",
        "(int, default 0) number of iterations over which to warmup lr")
    .Arg(
        "start_multiplier",
        "(float, default 0) starting multiplier for learning rate")
    .Arg(
        "end_multiplier",
        "(float, default 0) end multiplier for learning rate")
    .Arg(
        "multiplier",
        "(float, default 0.5) constant multiplier for learning rate")
    .Arg(
        "multiplier_1",
        "(float, default 1) start multiplier for learning rate")
    .Arg("multiplier_2", "(float, default 1) end multiplier for learning rate")
    .Arg(
        "sub_policy_num_iters",
        "(int array, default empty) number of iterations for each sub learning rate policy in composite policy")
    .Arg("m1", "")
    .Arg("n1", "")
    .Arg("m2", "")
    .Arg("n2", "")
    .Arg("m3", "")
    .Input(0, "input", "description needed")
    .Output(0, "output", "description needed")
    .DeviceInferenceFunction([](const OperatorDef& def) {
      return std::make_pair(
          std::vector<DeviceOption>{DeviceOption()},
          std::vector<DeviceOption>{def.device_option()});
    });

NO_GRADIENT(LearningRate);
}  // namespace caffe2
