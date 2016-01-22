#include "caffe2/utils/math.h"
#include "caffe2/operators/relu_op.h"

namespace caffe2 {

template <>
bool ReluOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_DCHECK_GT(X.size(), 0);
  Y->ReshapeLike(X);
  EigenVectorMap<float>(Y->mutable_data<float>(), X.size()) =
      ConstEigenVectorMap<float>(X.data<float>(), X.size()).cwiseMax(0.f);

  /* Naive implementation
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  for (int i = 0; i < X.size(); ++i) {
    Ydata[i] = std::max(Xdata[i], 0.f);
  }
  */
  return true;
}

template <>
bool ReluGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_DCHECK_GT(Y.size(), 0);
  CAFFE_DCHECK_EQ(dY.size(), Y.size());
  dX->ReshapeLike(Y);

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  #pragma omp parallel for
  for (int i = 0; i < Y.size(); ++i) {
    dXdata[i] = Ydata[i] > 0 ? dYdata[i] : 0;
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(Relu, ReluOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ReluGradient, ReluGradientOp<float, CPUContext>);

struct GetReluGradient : public GetGradientDefBase {
  vector<OperatorDef>* Create(const OperatorDef& def) override {
    return SingleGradientDef(
        def.type() + "Gradient", "",
        vector<string>{O(def, 0), GO(def, 0)},
        vector<string>{GI(def, 0)});
  }
};
REGISTER_GRADIENT(Relu, GetReluGradient);
REGISTER_GRADIENT(ReluFp16, GetReluGradient);

}  // namespace
}  // namespace caffe2
