/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
  # `linearWarmup`: uses `start_multiplier`, `num_iter`
  # `constantWarmup`: uses `multiplier`, `num_iter`

#### Optional:
* `stepsize`: defaults to 0
* `gamma`: defaults to 0
* `power`: defaults to 0
* `num_iter`: defaults to 0
* `start_multiplier`: defaults to 0
* `multiplier`: defaults to 0.5


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
    .Arg(
        "num_iter",
        "(int, default 0) number of iterations over which to warmup lr")
    .Arg(
        "start_multiplier",
        "(float, default 0) starting multiplier for learning rate")
    .Arg(
        "multiplier",
        "(float, default 0.5) constant multiplier for learning rate")
    .Input(0, "input", "description needed")
    .Output(0, "output", "description needed")
    .DeviceInferenceFunction([](const OperatorDef& def) {
      return std::make_pair(
          std::vector<DeviceOption>{DeviceOption()},
          std::vector<DeviceOption>{def.device_option()});
    });

NO_GRADIENT(LearningRate);
}  // namespace caffe2
