#include "caffe2/operators/mod_op.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <>
template <typename T>
bool ModOp<CPUContext>::DoRunWithType() {
  auto& data = Input(DATA);
  auto N = data.size();
  const auto* data_ptr = data.template data<T>();

  auto* output = Output(0);
  output->ResizeLike(Input(DATA));
  auto* output_ptr = output->template mutable_data<T>();

  for (auto i = 0; i < N; i++) {
    output_ptr[i] = data_ptr[i] % divisor_;
    if (output_ptr[i] && sign_follow_divisor_ &&
        ((output_ptr[i] > 0) != (divisor_ > 0))) {
      output_ptr[i] += divisor_;
    }
  }
  return true;
}

namespace {

REGISTER_CPU_OPERATOR(Mod, ModOp<CPUContext>);
OPERATOR_SCHEMA(Mod)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("divisor", "The divisor of the modulo operation. Must >= 1")
    .Arg(
        "sign_follow_divisor",
        "The sign of output follows Dividend if set to `false`. \
          Otherwise follows Divisor")
    .IdenticalTypeAndShape()
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Elementwise modulo operation. Each element in the output is the modulo result
of the corresponding elment in the input data. The divisor of the modulo is
provided by the operator argument `divisor`.
)DOC")
    .Input(0, "data", "input int32 or int64 data")
    .Output(0, "output", "output of data with modulo operation applied");

SHOULD_NOT_DO_GRADIENT(ModOp);
} // namespace
} // namespace caffe2
