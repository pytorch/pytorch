#ifndef CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_
#define CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// This is Caffe's InnerProductOp, with a name that fits its purpose better.
template <
    class Context,
    class Engine = DefaultEngine,
    bool TransposeWeight = true>
class FullyConnectedOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(this->template GetSingleArgument<int32_t>("axis", 1)),
        axis_w_(this->template GetSingleArgument<int32_t>("axis_w", 1)),
        float16_compute_(
            this->template GetSingleArgument<bool>("float16_compute", false)) {}
  ~FullyConnectedOp() {}

  template <
      typename T_X,
      typename T_W,
      typename T_B,
      typename T_Y,
      typename MATH>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const auto& W = Input(1);
    const auto& b = Input(2);
    auto* Y = Output(0);
    CAFFE_ENFORCE(b.dim() == 1, b.dim());
    // batch size
    const auto canonical_axis = X.canonical_axis_index(axis_);
    const auto M = X.size_to_dim(canonical_axis);
    const auto K = X.size_from_dim(canonical_axis);
    const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
    const int N = TransposeWeight ? W.size_to_dim(canonical_axis_w)
                                  : W.size_from_dim(canonical_axis_w);

    auto dimErrorString = [&]() {
      return c10::str(
          "Dimension mismatch: ",
          "X: ",
          X.sizes(),
          ", W: ",
          W.sizes(),
          ", b: ",
          b.sizes(),
          ", axis: ",
          axis_,
          ", M: ",
          M,
          ", N: ",
          N,
          ", K: ",
          K);
    };

    // Error checking
    CAFFE_ENFORCE(M == X.numel() / K, dimErrorString());
    CAFFE_ENFORCE(K == W.numel() / N, dimErrorString());
    CAFFE_ENFORCE(N == b.dim32(0), dimErrorString());
    CAFFE_ENFORCE(N == b.numel(), dimErrorString());

    Y_shape_cache_ = X.sizes().vec();
    // This is an invariant of canonical_axis, so we can DCHECK.
    DCHECK_LE(canonical_axis + 1, Y_shape_cache_.size());
    Y_shape_cache_.resize(canonical_axis + 1);
    Y_shape_cache_[canonical_axis] = N;
    Y->Resize(Y_shape_cache_);
    CAFFE_ENFORCE(M * N == Y->numel(), dimErrorString());

    if (X.numel() == 0) {
      // skip the rest of the computation if X is empty
      Y->template mutable_data<T_Y>();
      return true;
    }

    // default to FLOAT as math.h does.
    TensorProto::DataType math_type = TensorProto_DataType_FLOAT;
    if (fp16_type<MATH>()) {
      math_type = TensorProto_DataType_FLOAT16;
    }

    // W * x
    math::Gemm<T_X, Context, Engine>(
        CblasNoTrans,
        TransposeWeight ? CblasTrans : CblasNoTrans,
        M,
        N,
        K,
        1,
        X.template data<T_X>(),
        W.template data<T_W>(),
        0,
        Y->template mutable_data<T_Y>(),
        &context_,
        math_type);
    // Add bias term
    if (bias_multiplier_.numel() != M) {
      // If the helper bias multiplier is not M, reshape and fill it with one.
      bias_multiplier_.Resize(M);
      math::Set<T_B, Context>(
          M,
          convert::To<float, T_B>(1),
          bias_multiplier_.template mutable_data<T_B>(),
          &context_);
    }
    math::Gemm<T_B, Context, Engine>(
        CblasNoTrans,
        CblasNoTrans,
        M,
        N,
        1,
        1,
        bias_multiplier_.template data<T_B>(),
        b.template data<T_B>(),
        1,
        Y->template mutable_data<T_Y>(),
        &context_,
        math_type);
    return true;
  }

  bool RunOnDevice() override {
    return DoRunWithType<
        float, // X
        float, // W
        float, // B
        float, // Y
        float>(); // Math
  }

 protected:
  size_t axis_{1};
  size_t axis_w_{1};
  // A local vector to cache the output shape so we don't need to recreate
  // a vector object every time we run Run().
  vector<int64_t> Y_shape_cache_;
  Tensor bias_multiplier_{Context::GetDeviceType()};

  bool float16_compute_;
};

template <
    class Context,
    class Engine = DefaultEngine,
    bool TransposeWeight = true>
class FullyConnectedGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FullyConnectedGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(this->template GetSingleArgument<int32_t>("axis", 1)),
        axis_w_(this->template GetSingleArgument<int32_t>("axis_w", 1)),
        float16_compute_(
            this->template GetSingleArgument<bool>("float16_compute", false)) {}
  ~FullyConnectedGradientOp() {}

  template <
      typename T_X,
      typename T_W,
      typename T_DY,
      typename T_B,
      typename T_DX,
      typename T_DW,
      typename T_DB,
      typename MATH>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const auto& W = Input(1);
    const auto& dY = Input(2);
    // batch size
    const auto canonical_axis = X.canonical_axis_index(axis_);
    const int M = X.size_to_dim(canonical_axis);
    const int K = X.size_from_dim(canonical_axis);
    const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
    const int N = TransposeWeight ? W.size_to_dim(canonical_axis_w)
                                  : W.size_from_dim(canonical_axis_w);

    auto dimErrorString = [&]() {
      return c10::str(
          "Dimension mismatch: ",
          "X: ",
          X.sizes(),
          ", W: ",
          W.sizes(),
          ", dY: ",
          dY.sizes(),
          ", axis: ",
          axis_,
          ", M: ",
          M,
          ", N: ",
          N,
          ", K: ",
          K);
    };

    CAFFE_ENFORCE(M * K == X.numel(), dimErrorString());
    CAFFE_ENFORCE(K * N == W.numel(), dimErrorString());

    auto* dW = Output(0);
    auto* db = Output(1);
    dW->ResizeLike(W);
    db->Resize(N);

    if (X.numel() == 0) {
      // generate a zero blob for db and dW when X is empty
      math::Set<T_DB, Context>(
          db->numel(),
          convert::To<float, T_DB>(0),
          db->template mutable_data<T_DB>(),
          &context_);
      math::Set<T_DW, Context>(
          dW->numel(),
          convert::To<float, T_DW>(0),
          dW->template mutable_data<T_DW>(),
          &context_);

      if (OutputSize() == 3) {
        auto* dX = Output(2);
        dX->ResizeLike(X);
        dX->template mutable_data<T_DX>();
      }

      return true;
    }

    // default to FLOAT as math.h does.
    TensorProto::DataType math_type = TensorProto_DataType_FLOAT;
    if (fp16_type<MATH>()) {
      math_type = TensorProto_DataType_FLOAT16;
    }

    // Compute dW
    math::Gemm<T_DY, Context, Engine>(
        CblasTrans,
        CblasNoTrans,
        TransposeWeight ? N : K,
        TransposeWeight ? K : N,
        M,
        1,
        TransposeWeight ? dY.template data<T_DY>() : X.template data<T_X>(),
        TransposeWeight ? X.template data<T_X>() : dY.template data<T_DY>(),
        0,
        dW->template mutable_data<T_DW>(),
        &context_,
        math_type);
    if (bias_multiplier_.numel() != M) {
      // If the helper bias multiplier is not M, reshape and fill it
      // with one.
      bias_multiplier_.Resize(M);
      math::Set<T_B, Context>(
          M,
          convert::To<float, T_B>(1),
          bias_multiplier_.template mutable_data<T_B>(),
          &context_);
    }
    // Compute dB
    math::Gemv<T_DY, Context>(
        CblasTrans,
        M,
        N,
        1,
        dY.template data<T_DY>(),
        bias_multiplier_.template data<T_B>(),
        0,
        db->template mutable_data<T_DB>(),
        &context_);

    // Compute dX
    if (OutputSize() == 3) {
      auto* dX = Output(2);
      dX->ResizeLike(X);
      math::Gemm<T_DX, Context, Engine>(
          CblasNoTrans,
          TransposeWeight ? CblasNoTrans : CblasTrans,
          M,
          K,
          N,
          1,
          dY.template data<T_DY>(),
          W.template data<T_W>(),
          0,
          dX->template mutable_data<T_DX>(),
          &context_,
          math_type);
    }
    return true;
  }

  bool RunOnDevice() override {
    return DoRunWithType<
        float, //  X
        float, //  W
        float, // dY
        float, //  B
        float, // dX
        float, // dW
        float, // dB
        float>(); // Math
  }

 protected:
  size_t axis_{1};
  size_t axis_w_{1};
  Tensor bias_multiplier_{Context::GetDeviceType()};
  bool float16_compute_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FULLY_CONNECTED_OP_H_
