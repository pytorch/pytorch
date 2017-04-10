#ifndef CAFFE2_OPERATORS_REDUCTION_OPS_H_
#define CAFFE2_OPERATORS_REDUCTION_OPS_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SumElementsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  SumElementsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        average_(OperatorBase::GetSingleArgument<bool>("average", false)) {}
  SumElementsOp(const OperatorDef& operator_def, Workspace* ws, bool average)
      : Operator<Context>(operator_def, ws), average_(average) {}
  ~SumElementsOp() {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* sum = Output(0);
    sum->Resize(vector<TIndex>());
    T* data = sum->template mutable_data<T>();
    math::Sum<T, Context>(X.size(), X.template data<T>(), data, &context_);
    if (average_) {
      math::Scale<T, Context>(
          1,
          static_cast<T>(1.) / X.size(),
          sum->template data<T>(),
          data,
          &context_);
    }
    return true;
  }

 private:
  bool average_;
};

template <typename T, class Context>
class SumElementsGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  SumElementsGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        average_(OperatorBase::GetSingleArgument<bool>("average", false)) {}
  SumElementsGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws,
      bool average)
      : Operator<Context>(operator_def, ws), average_(average) {}
  ~SumElementsGradientOp() {}

  bool RunOnDevice() override;

 private:
  bool average_;
};

template <typename T, class Context>
class SumSqrElementsOp : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(SumSqrElementsOp)
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    bool average = OperatorBase::GetSingleArgument<bool>("average", false);
    auto& X = Input(0);
    auto* sum = Output(0);
    sum->Resize(vector<TIndex>());
    math::SumSqr<T, Context>(
        X.size(),
        X.template data<T>(),
        sum->template mutable_data<T>(),
        &context_);
    if (average) {
      math::Scale<T, Context>(
          1,
          static_cast<T>(1.) / X.size(),
          sum->template data<T>(),
          sum->template mutable_data<T>(),
          &context_);
    }
    return true;
  }
};

} // namespace caffe2

#endif
