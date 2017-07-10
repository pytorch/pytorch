#include "caffe2/operators/normalize_op.h"

#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
bool NormalizeOp<T, Context>::RunOnDevice() {
  auto& input = Input(0);
  auto* output = Output(0);
  auto m = input.dim32(input.ndim() - 1);
  auto n = input.size() / m;
  output->ResizeLike(input);
  ConstEigenMatrixMap<T> inputMat(input.template data<T>(), m, n);
  EigenMatrixMap<T> outputMat(output->template mutable_data<T>(), m, n);
  outputMat = inputMat.colwise().normalized();
  return true;
}

template <typename T, class Context>
bool NormalizeGradientOp<T, Context>::RunOnDevice() {
  auto& input = Input(INPUT);
  DCHECK_EQ(input.ndim(), 2);
  auto m = input.dim32(input.ndim() - 1);
  auto n = input.size() / m;
  Output(GRAD_IN)->ResizeLike(input);
  ConstEigenArrayMap<T> inputMat(input.template data<T>(), m, n);
  ConstEigenArrayMap<T> gradOutMat(Input(GRAD_OUT).template data<T>(), m, n);
  EigenArrayMap<T> gradInMat(Output(GRAD_IN)->template mutable_data<T>(), m, n);

  auto square = inputMat.square();
  auto norm = square.colwise().sum().sqrt();
  gradInMat = gradOutMat.rowwise() * norm.inverse() -
      ((inputMat.rowwise() / norm.pow(3)).rowwise() *
       (gradOutMat * inputMat).colwise().sum());

  return true;
}

REGISTER_CPU_OPERATOR(Normalize, NormalizeOp<float, CPUContext>);
OPERATOR_SCHEMA(Normalize).NumInputs(1).NumOutputs(1).SetDoc(R"DOC(
Given a matrix, apply L2-normalization along the last dimension.
)DOC");

REGISTER_CPU_OPERATOR(NormalizeGradient,
                      NormalizeGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(NormalizeGradient).NumInputs(2).NumOutputs(1);

class GetNormalizeGradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 1);
    return SingleGradientDef(
        "NormalizeGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Normalize, GetNormalizeGradient);

template <typename T, class Context>
void NormalizeL1Op<T, Context>::DoNormalize(
    const T* xData,
    T* yData,
    const int m,
    const int n,
    const int sf) {
  using InnerStride = Eigen::InnerStride<Eigen::Dynamic>;
  using StridedVec =
      Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, 0, InnerStride>;
  using ConstStridedVec =
      Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>, 0, InnerStride>;

  for (int i = 0; i < n; ++i) {
    auto base = (i / sf) * sf * m + (i % sf);
    ConstStridedVec xVec(xData + base, 1, m, InnerStride(sf));
    auto norm = xVec.template lpNorm<1>();
    if (norm != 0) {
      StridedVec yVec(yData + base, 1, m, InnerStride(sf));
      yVec = xVec / norm;
    }
  }
};

REGISTER_CPU_OPERATOR(NormalizeL1, NormalizeL1Op<float, CPUContext>);
OPERATOR_SCHEMA(NormalizeL1).NumInputs(1).NumOutputs(1).SetDoc(R"DOC(
Given a matrix, apply L1-normalization along the specified axis.
)DOC");

} // namespace caffe2
