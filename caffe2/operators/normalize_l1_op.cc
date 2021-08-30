#include "caffe2/operators/normalize_l1_op.h"

#include "caffe2/core/tensor.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

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
OPERATOR_SCHEMA(NormalizeL1)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("axis", "axis to normalize")
    .SetDoc(R"DOC(
Given a matrix, apply L1-normalization along the specified axis.
)DOC");

} // namespace caffe2
