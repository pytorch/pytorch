#ifndef CAFFE2_OPERATORS_NORMALIZE_OP_H_
#define CAFFE2_OPERATORS_NORMALIZE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

#define KEPS 1e-12f

namespace caffe2 {

template <typename T, class Context>
class NormalizeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit NormalizeOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    const auto& x = Input(0);

    const auto* xData = x.template data<T>();
    auto* y = Output(0, x.sizes(), at::dtype<T>());
    auto* yData = y->template mutable_data<T>();

    const auto canonical_axis = x.canonical_axis_index(
        this->template GetSingleArgument<int>("axis", -1));
    const int64_t m = x.dim(canonical_axis);
    const size_t n = x.numel() / m;
    const size_t sf = x.size_from_dim(canonical_axis + 1);
    DoNormalize(xData, yData, m, n, sf);
    return true;
  }

 private:
  const T kEps_ = KEPS;
  void DoNormalize(
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
      norm = std::max(norm, kEps_);
      StridedVec yVec(yData + base, 1, m, InnerStride(sf));
      yVec = xVec / norm;
    }
  }
};

template <typename T, class Context>
class NormalizeGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit NormalizeGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    const auto& x = Input(0);
    const auto& gOut = Input(GRAD_OUT);

    auto* gIn = Output(GRAD_IN, gOut.sizes(), at::dtype<T>());

    const auto* xData = x.template data<T>();
    const auto* gOutData = gOut.template data<T>();
    auto* gInData = gIn->template mutable_data<T>();

    const auto canonical_axis = x.canonical_axis_index(
        this->template GetSingleArgument<int>("axis", -1));
    const int m = x.dim32(canonical_axis);
    const int n = x.numel() / m;
    const int sf = x.size_from_dim(canonical_axis + 1);
    DoNormalize(xData, gOutData, gInData, m, n, sf);
    return true;
  }

 private:
  const T kEps_ = KEPS;
  void DoNormalize(
      const T* xData,
      const T* gOutData,
      T* gInData,
      const int m,
      const int n,
      const int sf);

  INPUT_TAGS(INPUT, GRAD_OUT);
  OUTPUT_TAGS(GRAD_IN);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_NORMALIZE_OP_H_
