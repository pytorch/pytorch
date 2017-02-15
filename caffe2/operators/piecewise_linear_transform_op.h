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
    binary_ = OperatorBase::GetSingleArgument<bool>("binary", false);
    CAFFE_ENFORCE(
        num_piece > 0,
        "No pieces specified, please specify pieces through args");
    range_ = SetPiecewiseLinearFunctionParameter("bounds", num_piece + 1);
    W_ = SetPiecewiseLinearFunctionParameter("slopes", num_piece);
    b_ = SetPiecewiseLinearFunctionParameter("intercepts", num_piece);

    CAFFE_ENFORCE_EQ(range_.size(), W_.size());
    CAFFE_ENFORCE_EQ(range_.size(), b_.size());

    if (binary_) {
      CAFFE_ENFORCE_EQ(range_.size(), 1);
      CAFFE_ENFORCE_EQ(W_.size(), 1);
      CAFFE_ENFORCE_EQ(b_.size(), 1);
    }
  }

  bool RunOnDevice() override {
    return binary_ ? TransformBinary() : TransformGeneral();
  }

 private:
  bool TransformGeneral() {
    auto& X = Input(0);
    auto* Y = Output(0);
    DCHECK_EQ(X.ndim(), 2);
    TIndex N = X.dim32(0);
    TIndex M = X.dim32(1);
    DCHECK_EQ(range_.size(), M);
    DCHECK_EQ(W_.size(), M);
    DCHECK_EQ(b_.size(), M);

    Y->ResizeLike(X);
    const auto* Xdata = X.template data<float>();
    float* Ydata = Y->template mutable_data<float>();

    for (TIndex j = 0; j < M; ++j) {
      for (TIndex i = 0; i < N; ++i) {
        Ydata[i * M + j] = Piecewise_Linear_Transform(
            Xdata[i * M + j], range_[j], W_[j], b_[j]);
      }
    }
    return true;
  }

  bool TransformBinary() {
    auto& X = Input(0);
    auto* Y = Output(0);
    DCHECK_EQ(X.ndim(), 2);
    TIndex N = X.dim32(0);
    TIndex M = X.dim32(1);
    CAFFE_ENFORCE_EQ(
        M, 2, "If binary is set to true, the input must be Nx2 tensor");
    Y->ResizeLike(X);
    const auto* Xdata = X.template data<float>();
    float* Ydata = Y->template mutable_data<float>();

    for (TIndex i = 0; i < N; ++i) {
      Ydata[i * M + 1] =
          Piecewise_Linear_Transform(Xdata[i * M + 1], range_[0], W_[0], b_[0]);
      Ydata[i * M] = 1.0f - Ydata[i * M + 1];
    }

    return true;
  }

  vector<vector<T>> SetPiecewiseLinearFunctionParameter(
      const string& arg,
      const int denom) {
    vector<vector<T>> param;
    vector<T> param_flat = OperatorBase::GetRepeatedArgument<T>(arg);
    CAFFE_ENFORCE_EQ(param_flat.size() % denom, 0);
    int num_dim = param_flat.size() / denom;
    CAFFE_ENFORCE_GT(num_dim, 0);
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
  bool binary_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_PIECEWISE_LINEAR_TRANSFORM_OP_H_
