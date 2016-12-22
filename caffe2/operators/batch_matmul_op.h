#ifndef CAFFE2_OPERATORS_MATMUL_OP_H_
#define CAFFE2_OPERATORS_MATMUL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context, class Engine = DefaultEngine>
class BatchMatMulOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BatchMatMulOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        trans_a_(OperatorBase::GetSingleArgument<int>("trans_a", 0)),
        trans_b_(OperatorBase::GetSingleArgument<int>("trans_b", 0)) {}
  ~BatchMatMulOp() {}

  bool RunOnDevice() override {
    const auto& A = Input(0);
    const auto& B = Input(1);
    auto* Y = Output(0);

    CAFFE_ENFORCE_EQ(A.ndim(), 3);
    CAFFE_ENFORCE_EQ(B.ndim(), 3);
    CAFFE_ENFORCE_EQ(A.dim32(0), B.dim32(0));

    int a_dim0, a_dim1, b_dim0, b_dim1;

    if (trans_a_) {
      a_dim0 = A.dim32(2);
      a_dim1 = A.dim32(1);
    } else {
      a_dim0 = A.dim32(1);
      a_dim1 = A.dim32(2);
    }

    if (trans_b_) {
      b_dim0 = B.dim32(2);
      b_dim1 = B.dim32(1);
    } else {
      b_dim0 = B.dim32(1);
      b_dim1 = B.dim32(2);
    }

    // Error checking
    CAFFE_ENFORCE(
        a_dim1 == b_dim0,
        "Dimension mismatch: ",
        trans_a_ ? "trans(A): " : "A: ",
        a_dim0,
        " ",
        a_dim1,
        trans_b_ ? ", trans(B): " : ", B: ",
        b_dim0,
        " ",
        b_dim1);

    Y->Resize(A.dim(0), a_dim0, b_dim1);

    if (!A.dim(0)) {
      Y->template mutable_data<T>(); // create output tensor
      return true;
    }

    // Y = A * B
    auto a_offset = A.size() / A.dim(0);
    auto b_offset = B.size() / B.dim(0);
    auto y_offset = a_dim0 * b_dim1;
    for (int i = 0; i < A.dim32(0); ++i) {
      math::Gemm<T, Context, Engine>(
          trans_a_ ? CblasTrans : CblasNoTrans,
          trans_b_ ? CblasTrans : CblasNoTrans,
          a_dim0,
          b_dim1,
          a_dim1,
          1,
          A.template data<T>() + a_offset * i,
          B.template data<T>() + b_offset * i,
          0,
          Y->template mutable_data<T>() + y_offset * i,
          &context_);
    }
    return true;
  }

 protected:
  bool trans_a_;
  bool trans_b_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MATMUL_OP_H_
