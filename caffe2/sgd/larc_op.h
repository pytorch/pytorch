#ifndef CAFFE2_OPERATORS_LARC_OP_H_
#define CAFFE2_OPERATORS_LARC_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class LarcOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LarcOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        offset_(OperatorBase::GetSingleArgument<float>("offset", 0.5)),
        trust_(OperatorBase::GetSingleArgument<float>("trust", 0.001)),
        lr_min_(OperatorBase::GetSingleArgument<float>("lr_min", 0.02)) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& dX = Input(1);
    CAFFE_ENFORCE(
        dX.size() == X.size(), "Gradient size doesn't match parameter size.");
    CAFFE_ENFORCE_GE(offset_, 0);
    CAFFE_ENFORCE_GE(trust_, 0);
    CAFFE_ENFORCE_GE(lr_min_, 0);

    auto& wd = Input(2);
    auto& lr_max = Input(3);
    auto* lr_rescaled = Output(0);
    lr_rescaled->Resize(vector<TIndex>{1});

    Compute(
        dX.size(),
        X.template data<T>(),
        dX.template data<T>(),
        wd.template data<T>(),
        lr_max.template data<T>(),
        offset_,
        trust_,
        lr_min_,
        lr_rescaled->template mutable_data<T>());

    return true;
  }

 private:
  void Compute(
      TIndex N,
      const T* X_data,
      const T* dX_data,
      const T* wd,
      const T* lr_max,
      T offset,
      T trust,
      T lr_min,
      T* lr_rescaled);

  T offset_;
  T trust_;
  T lr_min_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LARC_OP_H_
