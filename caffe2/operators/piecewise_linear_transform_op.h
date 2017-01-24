#ifndef CAFFE2_OPERATORS_PIECEWISE_LINEAR_TRANSFORM_OP_H_
#define CAFFE2_OPERATORS_PIECEWISE_LINEAR_TRANSFORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class PiecewiseLinearTransformOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  PiecewiseLinearTransformOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    int num_piece = OperatorBase::GetSingleArgument<int>("pieces", 0);
    CAFFE_ENFORCE(
        num_piece > 0,
        "No pieces specified, please specify pieces through args");
    range_ = SetPiecewiseLinearFunctionParameter("bounds", num_piece + 1);
    W_ = SetPiecewiseLinearFunctionParameter("slopes", num_piece);
    b_ = SetPiecewiseLinearFunctionParameter("intercepts", num_piece);
  }

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto* Y = Output(0);
    DCHECK_EQ(X.ndim(), 2);
    int N = X.dim32(0);
    int M = X.dim32(1);
    DCHECK_EQ(range_.size(), M);
    DCHECK_EQ(W_.size(), M);
    DCHECK_EQ(b_.size(), M);

    Y->ResizeLike(X);
    const auto* Xdata = X.template data<float>();
    float* Ydata = Y->template mutable_data<float>();

    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < N; ++i) {
        Ydata[i * M + j] = Piecewise_Linear_Transform(
            Xdata[i * M + j], range_[j], W_[j], b_[j]);
      }
    }
    return true;
  }

 protected:
  vector<vector<T>> SetPiecewiseLinearFunctionParameter(
      const string& arg,
      const int denom) {
    vector<vector<T>> param;
    vector<T> param_flat = OperatorBase::GetRepeatedArgument<T>(arg);
    int num_dim = param_flat.size() / denom;
    param.resize(num_dim);
    for (int i = 0; i < num_dim; i++) {
      param[i] = vector<T>(
          param_flat.begin() + i * denom, param_flat.begin() + (i + 1) * denom);
    }
    return param;
  }

  float Piecewise_Linear_Transform(
      const float x,
      std::vector<float>& range,
      std::vector<float>& W,
      std::vector<float>& b) {
    float y = 0;
    // deal with samples out of range
    // make it the same as the upper/lower bound value
    if (x <= range[0]) {
      y = W[0] * range[0] + b[0];
    } else if (x >= range[range.size() - 1]) {
      y = W.back() * range.back() + b.back();
    } else {
      std::vector<float>::iterator low_bound =
          std::lower_bound(range.begin(), range.end(), x);
      int range_idx = low_bound - range.begin() - 1;
      // compute the piecewise linear transformation as Y
      y = W[range_idx] * x + b[range_idx];
    }
    return y;
  }

 private:
  vector<vector<T>> range_;
  vector<vector<T>> W_;
  vector<vector<T>> b_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_PIECEWISE_LINEAR_TRANSFORM_OP_H_
