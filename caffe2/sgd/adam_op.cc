#include "adam_op.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Adam, AdamOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Adam)
    .NumInputs(6)
    .NumOutputs(3, 4)
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

Computes the Adam update (https://arxiv.org/abs/1412.6980) for an
input gradient and momentum parameters. Concretely, given inputs
(param, m1, m2, grad, lr, iters),

    t = iters + 1
    correction_multiplier = sqrt(1 - power(beta2, t)) /
      (1 - power(beta1, t))
    m1_o = (beta1 * m1) + (1 - beta1) * grad
    m2_o = (beta2 * m2) + (1 - beta2) * np.square(grad)
    grad_o = correction_multiplier * m1_o / \
        (sqrt(m2_o) + epsilon)
    param_o = param + lr * grad_o

and returns (param_o, m1_o, m2_o, grad_o), in which grad_o is an optional output

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
    .Output(3, "output_grad", "Optional Effective gradient")
    .Arg("beta1", "Default 0.9")
    .Arg("beta2", "Default 0.999")
    .Arg("epsilon", "Default 1e-5");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(SparseAdam, SparseAdamOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(SparseAdam)
    .NumInputs(7)
    .NumOutputs(3, 4)
    .EnforceInplace({{0, 0}, {1, 1}, {2, 2}})
    .DeviceInferenceFunction([](const OperatorDef& def) {
      auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);
      // ITER input lives on CPU
      in_dev[6] = DeviceOption();
      return std::make_pair(in_dev, out_dev);
    })
    .SetDoc(R"DOC(

    Computes the Adam Update for the sparse case.
    Given inputs (param, moment1, moment2, indices, grad, lr, iter), runs the dense
    Adam on (param, moment1[indices], momemnt2[indices], lr, iter) and returns
    (new_param, new_moment1, new_moment2) as in dense case.
    Adam can be customized as Rectified Adam (RAdam) by setting enableRAdam = true.

    )DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment_1", "First moment history")
    .Input(2, "moment_2", "Second moment history")
    .Input(3, "indices", "Sparse indices")
    .Input(4, "grad", "Gradient computed")
    .Input(5, "lr", "learning rate")
    .Input(6, "iter", "iteration number")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment_1", "Updated first moment")
    .Output(2, "output_moment_2", "Updated second moment")
    .Output(3, "output_grad", "Optional Effective gradient")
    .Arg("beta1", "Default 0.9")
    .Arg("beta2", "Default 0.999")
    .Arg("epsilon", "Default 1e-5")
    .Arg("enableRAdam", "Default false");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    RowWiseSparseAdam,
    RowWiseSparseAdamOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(RowWiseSparseAdam)
    .NumInputs(7)
    .NumOutputs(3, 4)
    .EnforceInplace({{0, 0}, {1, 1}, {2, 2}})
    .DeviceInferenceFunction([](const OperatorDef& def) {
      auto op_device =
          def.has_device_option() ? def.device_option() : DeviceOption();
      vector<DeviceOption> in_dev(def.input_size(), op_device);
      vector<DeviceOption> out_dev(def.output_size(), op_device);
      // ITER input lives on CPU
      in_dev[6] = DeviceOption();
      return std::make_pair(in_dev, out_dev);
    })
    .SetDoc(R"DOC(

    Computes a modified Adam Update for the sparse case.
    Given inputs (param, moment1, moment2, indices, grad, lr, iter), runs the
    Adam update on (param, moment1[indices], moment2[indices], lr, iter) and returns
    (new_param, new_moment1, new_moment2), where moment2 is a 1D tensor
    with length equal to the number of rows in param:
    shape(moment2) == shape(param)[0]. Each element of  moment2 is
    applied to an entire row of param, and the new moment2 values are
    calculated by averaging across the row.

    )DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment_1", "First moment history")
    .Input(2, "moment_2", "Second moment history")
    .Input(3, "indices", "Sparse indices")
    .Input(4, "grad", "Gradient computed")
    .Input(5, "lr", "learning rate")
    .Input(6, "iter", "iteration number")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment_1", "Updated first moment")
    .Output(2, "output_moment_2", "Updated second moment")
    .Output(3, "output_grad", "Optional Effective gradient")
    .Arg("beta1", "Default 0.9")
    .Arg("beta2", "Default 0.999")
    .Arg("epsilon", "Default 1e-5");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Adam);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(SparseAdam);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(RowWiseSparseAdam);
} // namespace caffe2
