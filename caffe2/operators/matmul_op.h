#ifndef CAFFE2_OPERATORS_MATMUL_OP_H_
#define CAFFE2_OPERATORS_MATMUL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context, class Engine = DefaultEngine>
class MatMulOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MatMulOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        trans_a_(OperatorBase::GetSingleArgument<int>("trans_a", 0)),
        trans_b_(OperatorBase::GetSingleArgument<int>("trans_b", 0)) {}
  ~MatMulOp() {}

  bool RunOnDevice() override {
    const auto& A = Input(0);
    const auto& B = Input(1);
    auto* Y = Output(0);

    CAFFE_ENFORCE(A.ndim() == 2, A.ndim());
    CAFFE_ENFORCE(B.ndim() == 2, B.ndim());

    int a_dim0, a_dim1, b_dim0, b_dim1;

    if (trans_a_) {
      a_dim0 = A.dim32(1);
      a_dim1 = A.dim32(0);
    } else {
      a_dim0 = A.dim32(0);
      a_dim1 = A.dim32(1);
    }

    if (trans_b_) {
      b_dim0 = B.dim32(1);
      b_dim1 = B.dim32(0);
    } else {
      b_dim0 = B.dim32(0);
      b_dim1 = B.dim32(1);
    }

    auto dimErrorString = [&]() {
      return MakeString(
          "Dimension mismatch: ",
          trans_a_ ? "trans(A): " : "A: ",
          a_dim0,
          " ",
          a_dim1,
          trans_b_ ? ", trans(B): " : ", B: ",
          b_dim0,
          " ",
          b_dim1);
    };
    // Error checking
    CAFFE_ENFORCE(a_dim1 == b_dim0, dimErrorString());

    Y_shape_cache_[0] = a_dim0;
    Y_shape_cache_[1] = b_dim1;
    Y->Resize(Y_shape_cache_);
    CAFFE_ENFORCE(a_dim0 * b_dim1 == Y->size(), dimErrorString());

    // Y = A * B
    math::Gemm<T, Context, Engine>(
        trans_a_ ? CblasTrans : CblasNoTrans,
        trans_b_ ? CblasTrans : CblasNoTrans,
        a_dim0,
        b_dim1,
        a_dim1,
        1,
        A.template data<T>(),
        B.template data<T>(),
        0,
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }

 protected:
  // A local vector to cache the output shape so we don't need to recreate
  // a vector object every time we run Run().
  vector<TIndex> Y_shape_cache_{0, 0};
  bool trans_a_;
  bool trans_b_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MATMUL_OP_H_
