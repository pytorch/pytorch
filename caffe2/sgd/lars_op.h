#ifndef CAFFE2_OPERATORS_LARS_OP_H_
#define CAFFE2_OPERATORS_LARS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class LarsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LarsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        offset_(OperatorBase::GetSingleArgument<float>("offset", 0.5)) {}

  bool RunOnDevice() override {
    auto& dX = Input(0);
    auto& momentum = Input(1);
    CAFFE_ENFORCE(
        dX.size() == momentum.size(),
        "Gradient size doesn't match momentum size.");
    CAFFE_ENFORCE_GE(offset_, 0);

    auto* lr_rescale = Output(0);
    lr_rescale->Resize(vector<TIndex>{1});

    Compute(
        dX.size(),
        dX.template data<T>(),
        momentum.template data<T>(),
        offset_,
        lr_rescale->template mutable_data<T>());

    return true;
  }

 private:
  void Compute(
      TIndex N,
      const T* dX_data,
      const T* momentum,
      T offset,
      T* lr_rescale_data);

  T offset_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LARS_OP_H_
