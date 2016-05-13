#include "caffe2/operators/l2_distance_op.h"

namespace caffe2 {

template<>
bool SquaredL2DistanceOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto* distance = Output(0);
  CAFFE_DCHECK_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_DCHECK_EQ(X.dim32(i), Y.dim32(i));
  }
  int N = X.dim32(0);
  int D = X.size() / X.dim32(0);
  distance->Reshape(vector<TIndex>{N});
  float* distance_data = distance->mutable_data<float>();
  for (int i = 0; i < N; ++i) {
    float Xscale, Yscale, cross;
    math::Dot<float, CPUContext>(
        D, X.data<float>(), X.data<float>(), &Xscale, &context_);
    math::Dot<float, CPUContext>(
        D, Y.data<float>(), Y.data<float>(), &Yscale, &context_);
    math::Dot<float, CPUContext>(
        D, X.data<float>(), Y.data<float>(), &cross, &context_);
    distance_data[i] = (Xscale + Yscale) / 2. - cross;
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(SquaredL2Distance,
                      SquaredL2DistanceOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SquaredL2DistanceGradient,
                      SquaredL2DistanceGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(SquaredL2Distance).NumInputs(2).NumOutputs(1);
OPERATOR_SCHEMA(SquaredL2DistanceGradient).NumInputs(3).NumOutputs(2);

class GetSquaredL2DistanceGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SquaredL2DistanceGradient", "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(0), GI(1)});
  }
};
REGISTER_GRADIENT(SquaredL2Distance, GetSquaredL2DistanceGradient);
}  // namespace
}  // namespace caffe2
