#include "caffe2/operators/normalize_op.h"

#include "caffe2/core/tensor.h"

namespace caffe2 {

template <typename T, class Context>
void NormalizeOp<T, Context>::DoNormalize(
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
    auto norm = xVec.template lpNorm<2>();
    if (norm != 0) {
      StridedVec yVec(yData + base, 1, m, InnerStride(sf));
      yVec = xVec / norm;
    }
  }
};

template <typename T, class Context>
void NormalizeGradientOp<T, Context>::DoNormalize(
    const T* xData,
    const T* gOutData,
    T* gInData,
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
    ConstStridedVec gOutVec(gOutData + base, 1, m, InnerStride(sf));

    auto row_sum = xVec.dot(gOutVec);
    auto row_norm = xVec.template lpNorm<2>();
    auto row_norm_3 = pow(row_norm, 3);
    if (row_norm != 0) {
      StridedVec gInVec(gInData + base, 1, m, InnerStride(sf));
      gInVec = (gOutVec / row_norm) - ((xVec / row_norm_3) * row_sum);
    }
  }
};

REGISTER_CPU_OPERATOR(Normalize, NormalizeOp<float, CPUContext>);
OPERATOR_SCHEMA(Normalize)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("axis", "axis to normalize")
    .SetDoc(R"DOC(
Given a matrix, apply L2-normalization along the specified dimension.
)DOC")
    .IdenticalTypeAndShape();

REGISTER_CPU_OPERATOR(
    NormalizeGradient,
    NormalizeGradientOp<float, CPUContext>);
OPERATOR_SCHEMA(NormalizeGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .Arg("axis", "axis to normalize");

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

} // namespace caffe2
