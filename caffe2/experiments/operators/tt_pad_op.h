#ifndef CAFFE2_OPERATORS_TT_PAD_OP_H_
#define CAFFE2_OPERATORS_TT_PAD_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context, class Engine = DefaultEngine>
class TTPadOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  TTPadOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scale_(OperatorBase::GetSingleArgument<TIndex>("scale", 0)) {
    CAFFE_ENFORCE(
        OperatorBase::HasArgument("scale"), "Argument `scale` is missing.");
  }

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* X_pad = Output(0);
    CAFFE_ENFORCE(&X == X_pad);

    CAFFE_ENFORCE(X.ndim() == 2, X.ndim());

    auto X_dim0 = X.dim(0);
    auto X_dim1 = X.dim(1);

    auto* X_orig_dim0 = Output(1);
    X_orig_dim0->Resize(1);
    *X_orig_dim0->template mutable_data<TIndex>() = X_dim0;

    if (X_dim0 % scale_ != 0) {
      TIndex padded_dim0 = (X_dim0 / scale_ + 1) * scale_;
      auto dim0_diff = padded_dim0 - X_dim0;
      // set growthPct to the upper bound percentage: (100 * scale_ / X_dim0)
      X_pad->template Extend(dim0_diff, 100 * scale_ / X_dim0, &context_);

      auto* X_pad_data = X_pad->template mutable_data<T>();
      TIndex X_size = X_dim0 * X_dim1;
      memset(X_pad_data + X_size, 0, dim0_diff * X_dim1 * sizeof(T));
    }

    return true;
  }

 protected:
  TIndex scale_;
};

template <typename T, class Context, class Engine = DefaultEngine>
class TTPadGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  TTPadGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    const auto& G = Input(0);
    auto* output = Output(0);
    CAFFE_ENFORCE(&G == output);

    auto old_dim0 = *Input(1).template data<TIndex>();
    auto new_dim0 = G.dim(0);
    auto dim1 = G.dim(1);

    if (old_dim0 < new_dim0) {
      output->Shrink(old_dim0);
    }

    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TT_PAD_OP_H_
