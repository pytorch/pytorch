#include "caffe2/operators/minmax_ops.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(MaxGradient, MaxGradientOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MinGradient, MinGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(MaxGradient).NumInputs(3, INT_MAX).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(MinGradient).NumInputs(3, INT_MAX).NumOutputs(1, INT_MAX);

template <typename T, class Context>
bool SelectGradientOpBase<T, Context>::RunOnDevice() {
  auto& output = Input(0);
  auto& grad_output = Input(1);
  const int kInputStartOffset = 2;

  const T* data = output.template data<T>();
  ConstEigenArrayMap<T> output_array(
      output.template data<T>(), 1, output.size());
  ConstEigenArrayMap<T> grad_out_array(
      grad_output.template data<T>(), 1, grad_output.size());

  for (int i = 0; i < OutputSize(); i++) {
    auto& input = Input(i + kInputStartOffset);
    ConstEigenArrayMap<T> input_array(
        input.template data<T>(), 1, input.size());

    auto* grad_input = Output(i);
    grad_input->ResizeLike(input);
    EigenArrayMap<T> grad_in_array(
        grad_input->template mutable_data<T>(), 1, grad_input->size());
    grad_in_array = grad_out_array *
        input_array.cwiseEqual(output_array).template cast<T>();
  }
  return true;
}

class GetMaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    auto gradInputs = vector<string>();
    auto inputs = vector<string>{O(0), GO(0)};
    for (int i = 0; i < def_.input_size(); i++) {
      gradInputs.push_back(GI(i));
      inputs.push_back(I(i));
    }
    return SingleGradientDef("MaxGradient", "", inputs, gradInputs);
  }
};
REGISTER_GRADIENT(Max, GetMaxGradient);

class GetMinGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    auto gradInputs = vector<string>();
    auto inputs = vector<string>{O(0), GO(0)};
    for (int i = 0; i < def_.input_size(); i++) {
      gradInputs.push_back(GI(i));
      inputs.push_back(I(i));
    }
    return SingleGradientDef("MinGradient", "", inputs, gradInputs);
  }
};
REGISTER_GRADIENT(Min, GetMinGradient);

} // namespace caffe2
