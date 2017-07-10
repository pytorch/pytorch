#ifndef CAFFE2_OPERATORS_NORMALIZE_OP_H_
#define CAFFE2_OPERATORS_NORMALIZE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class NormalizeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  NormalizeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}

  bool RunOnDevice() override;
};

template <typename T, class Context>
class NormalizeGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  NormalizeGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}

  bool RunOnDevice() override;

 private:
  INPUT_TAGS(INPUT, GRAD_OUT);
  OUTPUT_TAGS(GRAD_IN);
};

template <typename T, class Context>
class NormalizeL1Op final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(NormalizeL1Op)

  bool RunOnDevice() override {
    const auto& x = Input(0);
    auto* y = Output(0);
    const auto* xData = x.template data<T>();
    y->ResizeLike(x);
    auto* yData = y->template mutable_data<T>();

    const auto canonical_axis = x.canonical_axis_index(
        OperatorBase::GetSingleArgument<int>("axis", -1));
    const int m = x.dim32(canonical_axis);
    const int n = x.size() / m;
    const int sf = x.size_from_dim(canonical_axis + 1);
    DoNormalize(xData, yData, m, n, sf);
    return true;
  }

 private:
  void
  DoNormalize(const T* xData, T* yData, const int m, const int n, const int sf);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_NORMALIZE_OP_H_
