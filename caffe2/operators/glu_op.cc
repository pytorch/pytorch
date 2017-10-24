#include <math.h>

#include "caffe2/operators/glu_op.h"

namespace caffe2 {

namespace {
float sigmoid(const float x) {
  if (x >= 0) {
    return 1. / (1. + exp(-x));
  } else {
    const float exp_x = exp(x);
    return exp_x / (1 + exp_x);
  }
}
} // namespace

template <>
void GluOp<float, CPUContext>::ComputeGlu(
    const int M,
    const int N,
    const float* Xdata,
    float* Ydata) {
  const int xOffset = 2 * N;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      Ydata[i * N + j] =
          Xdata[i * xOffset + j] * sigmoid(Xdata[i * xOffset + j + N]);
    }
  }
}

OPERATOR_SCHEMA(Glu)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
  Applies gated linear unit to the input Tensor X. The output Y is half the size
  of the input X, so if the shape of X is [d1, d2, ..., N] shape of Y will be
   [d1, d2, ..., dn/2] and Y(:dn-1, i) = GLU(X(:dn-1, i), X(:dn-1, i+N/2)) =
   X(dn-1, i) * sigmoid(X(dn-1, i+N/2))
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D output tensor");

REGISTER_CPU_OPERATOR(Glu, GluOp<float, CPUContext>);
} // namespace caffe2
