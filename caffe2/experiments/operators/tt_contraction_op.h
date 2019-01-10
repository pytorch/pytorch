#ifndef CAFFE2_OPERATORS_TT_CONTRACTION_OP_H_
#define CAFFE2_OPERATORS_TT_CONTRACTION_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context, class Engine = DefaultEngine>
class TTContractionOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  TTContractionOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        K_(OperatorBase::GetSingleArgument<TIndex>("K", 0)),
        M_(OperatorBase::GetSingleArgument<TIndex>("M", 0)),
        N_(OperatorBase::GetSingleArgument<TIndex>("N", 0)) {
    CAFFE_ENFORCE(OperatorBase::HasArgument("K"), "Argument `K` is missing.");
    CAFFE_ENFORCE(OperatorBase::HasArgument("M"), "Argument `M` is missing.");
    CAFFE_ENFORCE(OperatorBase::HasArgument("N"), "Argument `N` is missing.");
  }

  bool RunOnDevice() override {
    const auto& A = Input(0);
    const auto& B = Input(1);
    auto* C = Output(0);

    CAFFE_ENFORCE(A.ndim() == 2, A.ndim());

    TIndex A_size = A.size_from_dim(0);
    TIndex B_size = B.size_from_dim(0);

    CAFFE_ENFORCE(
        K_ * M_ == A_size,
        "Argument `K` and `M` do not agree with the size of A.");

    CAFFE_ENFORCE(
        B_size % (K_ * N_) == 0,
        "Argument `K` and `N` do not agree with the size of B.");

    TIndex D_ = B_size / (K_ * N_);

    TIndex C_size = D_ * M_ * N_;
    C->Resize(vector<TIndex>{C_size});

    TIndex B_stride = K_ * N_;
    TIndex C_stride = M_ * N_;

    const T* A_data = A.template data<T>();
    const T* B_data = B.template data<T>();
    T* C_data = C->template mutable_data<T>();

    for (TIndex B_index = 0; B_index < B_size; B_index += B_stride) {
      math::Gemm<T, Context, Engine>(
          CblasTrans,
          CblasNoTrans,
          M_, N_, K_, 1,
          A_data,
          B_data + B_index,
          0,
          C_data,
          &context_);
      C_data += C_stride;
    }

    return true;
  }

 protected:
  TIndex K_;
  TIndex M_;
  TIndex N_;
};

template <typename T, class Context, class Engine = DefaultEngine>
class TTContractionGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  TTContractionGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        K_(OperatorBase::GetSingleArgument<TIndex>("K", 0)),
        M_(OperatorBase::GetSingleArgument<TIndex>("M", 0)),
        N_(OperatorBase::GetSingleArgument<TIndex>("N", 0)) {}

  bool RunOnDevice() override {
    const auto& G = Input(0);
    const auto& A = Input(1);
    const auto& B = Input(2);
    auto* dA = Output(0);
    auto* dB = Output(1);

    TIndex G_size = G.size_from_dim(0);
    TIndex D_ = G_size / (M_ * N_);

    TIndex dB_size = D_ * K_ * N_;

    dA->Resize(A.dims());
    dB->Resize(B.dims());

    TIndex B_stride = K_ * N_;
    TIndex G_stride = M_ * N_;

    const T* G_data = G.template data<T>();
    const T* A_data = A.template data<T>();
    const T* B_data = B.template data<T>();

    T* dA_data = dA->template mutable_data<T>();
    T* dB_data = dB->template mutable_data<T>();

    const T* G_ptr = G_data;
    for (TIndex B_index = 0; B_index < dB_size; B_index += B_stride) {
      math::Gemm<T, Context, Engine>(
          CblasNoTrans,
          CblasTrans,
          K_, M_, N_, 1,
          B_data + B_index,
          G_ptr,
          B_index == 0 ? 0 : 1,
          dA_data,
          &context_);
      G_ptr += G_stride;
    }

    G_ptr = G_data;
    for (TIndex B_index = 0; B_index < dB_size; B_index += B_stride) {
      math::Gemm<T, Context, Engine>(
          CblasNoTrans,
          CblasNoTrans,
          K_, N_, M_, 1,
          A_data,
          G_ptr,
          0,
          dB_data + B_index,
          &context_);
      G_ptr += G_stride;
    }

    return true;
  }

 protected:
  TIndex K_;
  TIndex M_;
  TIndex N_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TT_CONTRACTION_OP_H_
