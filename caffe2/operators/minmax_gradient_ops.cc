#include "caffe2/operators/minmax_ops.h"

#include <string>
#include <vector>

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <typename T, class Context>
bool SelectGradientOpBase<T, Context>::RunOnDevice() {
  const auto& Y = Input(0);
  const auto& dY = Input(1);
  const int N = Y.numel();
  ConstEigenVectorArrayMap<T> Y_arr(Y.template data<T>(), N);
  ConstEigenVectorArrayMap<T> dY_arr(dY.template data<T>(), N);
  for (int i = 0; i < OutputSize(); i++) {
    const auto& Xi = Input(i + 2);
    auto* dXi = Output(i, Xi.sizes(), at::dtype<T>());
    ConstEigenVectorArrayMap<T> Xi_arr(Xi.template data<T>(), N);
    EigenVectorArrayMap<T> dXi_arr(dXi->template mutable_data<T>(), N);
    dXi_arr = (Xi_arr == Y_arr).template cast<T>() * dY_arr;
  }
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(MaxGradient, MaxGradientOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(MinGradient, MinGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MaxGradient).NumInputs(3, INT_MAX).NumOutputs(1, INT_MAX);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(MinGradient).NumInputs(3, INT_MAX).NumOutputs(1, INT_MAX);

namespace {

class GetMaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    std::vector<std::string> inputs = {O(0), GO(0)};
    std::vector<std::string> grad_inputs;
    for (int i = 0; i < def_.input_size(); ++i) {
      inputs.push_back(I(i));
      grad_inputs.push_back(GI(i));
    }
    return SingleGradientDef("MaxGradient", "", inputs, grad_inputs);
  }
};

class GetMinGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    std::vector<std::string> inputs = {O(0), GO(0)};
    std::vector<std::string> grad_inputs;
    for (int i = 0; i < def_.input_size(); ++i) {
      inputs.push_back(I(i));
      grad_inputs.push_back(GI(i));
    }
    return SingleGradientDef("MinGradient", "", inputs, grad_inputs);
  }
};

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Max, GetMaxGradient);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Min, GetMinGradient);

} // namespace caffe2
