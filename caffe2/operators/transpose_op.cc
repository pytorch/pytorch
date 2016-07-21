#include "caffe2/operators/transpose_op.h"

namespace caffe2 {

#define COMPILE_TIME_MAX_TRANSPOSE_DIMS 10

template <>
template <typename T>
bool TransposeOp<CPUContext>::DoRunWithType() {
  int from_inds[COMPILE_TIME_MAX_TRANSPOSE_DIMS] = {0};
  const auto& input = Input(0);
  auto* output = Output(0);
  size_t count = input.size();
  const auto& from_counts = input.dims();
  const auto& to_counts = output->dims();
  int num_axes = from_counts.size();
  const T* from_data = input.template data<T>();
  T* to_data = output->template mutable_data<T>();
  for (size_t index = 0; index < count; index++) {
    size_t from_index = index, to_index = 0;
    for (int i = num_axes - 1; i >= 0; --i) {
      from_inds[i] = from_index % from_counts[i];
      from_index = from_index / from_counts[i];
    }
    for (int i = 0; i < num_axes - 1; ++i) {
      to_index = (to_index + from_inds[axes_[i]]) * to_counts[i + 1];
    }
    to_index += from_inds[axes_[num_axes - 1]];
    *(to_data + to_index) = *(from_data + index);
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(Transpose, TransposeOp<CPUContext>);

OPERATOR_SCHEMA(Transpose)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Transpose the input tensor similar to numpy.transpose. For example, when
axes=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape
will be (2, 1, 3).
)DOC")
    .Arg(
        "axes",
        "A list of integers. By default, reverse the dimensions, "
        "otherwise permute the axes according to the values given.")
    .Input(0, "data", "An input tensor.")
    .Output(0, "transposed", "Transposed output.");

class GetTransposeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  // We will create our own arguments.
  bool CopyArguments() const override {
    return false;
  }
  vector<OperatorDef> GetGradientDefs() override {
    auto ops = SingleGradientDef(
        "Transpose", "", vector<string>{GO(0)}, vector<string>{GI(0)});
    ops[0].mutable_arg()->CopyFrom(Def().arg());
    if (HasArgument(Def(), "axes")) {
      // If axes is specified, we will need to figure out the inverse index.
      const Argument& old_axes = GetArgument(Def(), "axes");
      const int axes_size = old_axes.ints_size();
      Argument* new_arg = GetMutableArgument("axes", false, &ops[0]);
      for (int i = 0; i < axes_size; ++i) {
        new_arg->set_ints(old_axes.ints(i), i);
      }
    }
    return ops;
  }
};
REGISTER_GRADIENT(Transpose, GetTransposeGradient);
} // namespace
} // namespace caffe2
