#include "caffe2/operators/softmax_op.h"

namespace caffe2 {

// Implementation for the CPU context.
template <>
bool SoftmaxOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_DCHECK_EQ(X.ndim(), 2);
  int N = X.dim32(0);
  int D = X.dim32(1);
  Y->ReshapeLike(X);
  float* Ydata = Y->mutable_data<float>();
  // First, get scales
  if (scale_.size() != N) {
    scale_.Reshape(vector<TIndex>{N});
  }
  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Reshape(vector<TIndex>{D});
    math::Set<float, CPUContext>(D, 1.f, sum_multiplier_.mutable_data<float>(),
                                 &context_);
  }
  math::RowwiseMax<float, CPUContext>(N, D, X.data<float>(), scale_.mutable_data<float>(),
                                      &context_);
  // Put the intermediate result X - max(X) into Y
  context_.template Copy<float, CPUContext, CPUContext>(
      X.size(), X.data<float>(), Ydata);
  // Subtract the scale
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, N, D, 1,
      -1, scale_.data<float>(), sum_multiplier_.data<float>(), 1,
      Ydata, &context_);
  // Exponentiation
  math::Exp<float, CPUContext>(Y->size(), Ydata, Ydata,
                               &context_);
  math::Gemv<float, CPUContext>(CblasNoTrans, N, D, 1, Ydata,
                                sum_multiplier_.data<float>(), 0,
                                scale_.mutable_data<float>(), &context_);
  // Do division
  // TODO(Yangqing): maybe implement it more beautifully?
  const float* scale = scale_.data<float>();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      Ydata[i * D + j] /= scale[i];
    }
  }
  return true;
}

// Implementation for the CPU context.
template <>
bool SoftmaxGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_DCHECK_EQ(Y.ndim(), 2);
  int N = Y.dim32(0);
  int D = Y.dim32(1);
  CAFFE_DCHECK_EQ(dY.dim32(0), N);
  CAFFE_DCHECK_EQ(dY.dim32(1), D);
  // First, get scales
  if (scale_.size() != N) {
    scale_.Reshape(vector<TIndex>{N});
  }
  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Reshape(vector<TIndex>{D});
    math::Set<float, CPUContext>(D, 1.f, sum_multiplier_.mutable_data<float>(),
                                 &context_);
  }
  dX->Reshape(vector<TIndex>{N, D});
  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  context_.Copy<float, CPUContext, CPUContext>(Y.size(), dYdata, dXdata);
  float* scaledata = scale_.mutable_data<float>();
  for (int i = 0; i < N; ++i) {
    math::Dot<float, CPUContext>(D, Ydata + i * D, dYdata + i * D,
                                 scaledata + i, &context_);
  }
  math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, N, D, 1, -1,
                                scaledata, sum_multiplier_.data<float>(), 1,
                                dXdata, &context_);
  math::Mul<float, CPUContext>(Y.size(), dXdata, Ydata, dXdata,
                               &context_);
  return true;
}

namespace {
REGISTER_CPU_OPERATOR(Softmax, SoftmaxOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SoftmaxGradient, SoftmaxGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Softmax).NumInputs(1).NumOutputs(1);
// Input: Y, dY. Output: dX
OPERATOR_SCHEMA(SoftmaxGradient).NumInputs(2).NumOutputs(1);

class GetSoftmaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient", "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Softmax, GetSoftmaxGradient);
REGISTER_GRADIENT(SoftmaxFp16, GetSoftmaxGradient);

}  // namespace
}  // namespace caffe2
