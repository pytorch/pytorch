#include "caffe2/sgd/decay_adagrad_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(DecayAdagrad, DecayAdagradOp<float, CPUContext>);
OPERATOR_SCHEMA(DecayAdagrad)
    .NumInputs(6)
    .NumOutputs(3)
    .AllowInplace({{0, 0}, {1, 1}, {2, 2}})
    .DeviceInferenceFunction([](const OperatorDef& def) {
      auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);
      // ITER input lives on CPU
      in_dev[5] = DeviceOption();
      return std::make_pair(in_dev, out_dev);
    })
    .SetDoc(R"DOC(

Computes the DecayAdagrad update for an
input gradient and momentum parameters. Concretely, given inputs
(param, m1, m2, c, grad, lr, iters),

    t = iters + 1
    m1_o = (beta1 * m1) + (1 - beta1) * grad
    m2_o = m2 + np.square(grad)
    c = 1.0 or (1 - power(beta1, t))
    grad_o = m1_o / c / (sqrt(m2_o) + epsilon)
    param_o = param + lr * (grad_o + weight_decay * param)

and returns (param_o, m1_o, m2_o)

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment_1", "First moment history")
    .Input(2, "moment_2", "Second moment history")
    .Input(3, "grad", "Gradient computed")
    .Input(4, "lr", "learning rate")
    .Input(5, "iter", "iteration number")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment_1", "Updated first moment")
    .Output(2, "output_moment_2", "Updated second moment")
    .Arg("beta1", "Default 0.9")
    .Arg("beta2", "Default 0.999")
    .Arg("epsilon", "Default 1e-5")
    .Arg("weight_decay", "Default 0.0")
    .Arg("bias_correction_first", "Default True");


SHOULD_NOT_DO_GRADIENT(DecayAdagrad);
} // namespace caffe2
