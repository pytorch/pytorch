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
  template <class... Args>
  explicit MatMulOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        axis_a_(this->template GetSingleArgument<int>("axis_a", 1)),
        axis_b_(this->template GetSingleArgument<int>("axis_b", 1)),
        trans_a_(this->template GetSingleArgument<int>("trans_a", 0)),
        trans_b_(this->template GetSingleArgument<int>("trans_b", 0)) {}
  ~MatMulOp() override {}

  bool RunOnDevice() override {
    const auto& A = Input(0);
    const auto& B = Input(1);

    const auto canonical_axis_a = A.canonical_axis_index(axis_a_);
    const auto canonical_axis_b = B.canonical_axis_index(axis_b_);
    int A_dim0 = A.size_to_dim(canonical_axis_a);
    int A_dim1 = A.size_from_dim(canonical_axis_a);
    int B_dim0 = B.size_to_dim(canonical_axis_b);
    int B_dim1 = B.size_from_dim(canonical_axis_b);

    int a_dim0, a_dim1, b_dim0, b_dim1;

    if (trans_a_) {
      a_dim0 = A_dim1;
      a_dim1 = A_dim0;
    } else {
      a_dim0 = A_dim0;
      a_dim1 = A_dim1;
    }

    if (trans_b_) {
      b_dim0 = B_dim1;
      b_dim1 = B_dim0;
    } else {
      b_dim0 = B_dim0;
      b_dim1 = B_dim1;
    }

    auto dimErrorString = [&]() {
      return c10::str(
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
    auto* Y = Output(0, Y_shape_cache_, at::dtype<T>());
    CAFFE_ENFORCE(a_dim0 * b_dim1 == Y->numel(), dimErrorString());
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

    if (InputSize() == 3) {
      // In gradient op, resize to input
      Y->ResizeLike(Input(2));
    }
    return true;
  }

 protected:
  // A local vector to cache the output shape so we don't need to recreate
  // a vector object every time we run Run().
  vector<int64_t> Y_shape_cache_{0, 0};
  int axis_a_{1};
  int axis_b_{1};
  bool trans_a_;
  bool trans_b_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MATMUL_OP_H_
