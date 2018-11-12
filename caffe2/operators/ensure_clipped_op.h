#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class EnsureClippedOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  EnsureClippedOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        min_(std::numeric_limits<T>::lowest()),
        max_(std::numeric_limits<T>::max()) {
    if (HasArgument("min")) {
      min_ = static_cast<T>(this->template GetSingleArgument<float>("min", 0));
    }
    if (HasArgument("max")) {
      max_ = static_cast<T>(this->template GetSingleArgument<float>("max", 0));
    }
  }

  bool RunOnDevice() override {
    if (InputSize() > INDICES) {
      // spares gradient, selective checking clipping
      CAFFE_ENFORCE_EQ(
          Input(PARAM).size_from_dim(1),
          Input(GRAD).size_from_dim(Input(INDICES).dim()));
      return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
          this, Input(INDICES));
    } else {
      auto& X = Input(PARAM);
      auto* Y = Output(OUTPUT_PARAM);
      Y->ResizeLike(X);
      EigenVectorMap<float>(Y->template mutable_data<float>(), Y->numel()) =
          ConstEigenVectorMap<float>(X.template data<float>(), X.numel())
              .cwiseMax(min_)
              .cwiseMin(max_);
      return true;
    }
  }

  template <typename SIndex>
  bool DoRunWithType();

 protected:
  T min_;
  T max_;
  INPUT_TAGS(PARAM, INDICES, GRAD);
  OUTPUT_TAGS(OUTPUT_PARAM);
};

} // namespace caffe2
