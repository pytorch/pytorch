#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

void SoftmaxCPU(
    CPUContext& context,
    const int N,
    const int D,
    const Tensor<CPUContext>& X,
    float* Ydata,
    Tensor<CPUContext>& scale,
    Tensor<CPUContext>& sum_multiplier) {
  math::RowwiseMax<float, CPUContext>(
      N, D, X.data<float>(), scale.mutable_data<float>(), &context);
  // Put the intermediate result X - max(X) into Y
  context.template Copy<float, CPUContext, CPUContext>(
      X.size(), X.data<float>(), Ydata);
  // Subtract the max (for nomuerical reasons)
  math::Gemm<float, CPUContext>(
      CblasNoTrans,
      CblasNoTrans,
      N,
      D,
      1,
      -1,
      scale.data<float>(),
      sum_multiplier.data<float>(),
      1,
      Ydata,
      &context);
  // Exponentiation
  math::Exp<float, CPUContext>(N * D, Ydata, Ydata, &context);
  math::Gemv<float, CPUContext>(
      CblasNoTrans,
      N,
      D,
      1,
      Ydata,
      sum_multiplier.data<float>(),
      0,
      scale.mutable_data<float>(),
      &context);
  // Do division
  // TODO(Yangqing): maybe implement it more beautifully?
  const float* s = scale.data<float>();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      Ydata[i * D + j] /= s[i];
    }
  }
}

} // namespace caffe2
