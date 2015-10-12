#include "caffe2/operators/l2_distance_op.h"

namespace caffe2 {

template<>
bool SquaredL2DistanceOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto* distance = Output(0);
  CAFFE_DCHECK_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_DCHECK_EQ(X.dim(i), Y.dim(i));
  }
  int N = X.dim(0);
  int D = X.size() / X.dim(0);
  distance->Reshape(std::vector<int>{N});
  float* distance_data = distance->mutable_data<float>();
  for (int i = 0; i < N; ++i) {
    float Xscale, Yscale, cross;
    math::Dot<float, CPUContext>(
        D, X.data<float>(), X.data<float>(), &Xscale, &device_context_);
    math::Dot<float, CPUContext>(
        D, Y.data<float>(), Y.data<float>(), &Yscale, &device_context_);
    math::Dot<float, CPUContext>(
        D, X.data<float>(), Y.data<float>(), &cross, &device_context_);
    distance_data[i] = (Xscale + Yscale) / 2. - cross;
  }
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(SquaredL2Distance,
                      SquaredL2DistanceOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SquaredL2DistanceGradient,
                      SquaredL2DistanceGradientOp<float, CPUContext>);

}
}  // namespace caffe2
