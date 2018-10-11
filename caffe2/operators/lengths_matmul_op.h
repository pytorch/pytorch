#ifndef CAFFE2_OPERATORS_LENGTHS_MATMUL_OP_H_
#define CAFFE2_OPERATORS_LENGTHS_MATMUL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
template <typename T, class Context, class Engine = DefaultEngine>
class LengthsMatMulOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  LengthsMatMulOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        trans_a_(OperatorBase::GetSingleArgument<int>("trans_a", 0)),
        trans_b_(OperatorBase::GetSingleArgument<int>("trans_b", 1)) {}

  ~LengthsMatMulOp() {}

  bool RunOnDevice() override {
    auto& A = Input(0);
    auto& lengthsA = Input(1);
    auto& B = Input(2);
    auto& lengthsB = Input(3);

    CAFFE_ENFORCE(A.ndim() <= 2);
    CAFFE_ENFORCE(B.ndim() == 2);
    CAFFE_ENFORCE(lengthsA.ndim() == 1);
    CAFFE_ENFORCE(lengthsB.ndim() == 1);
    CAFFE_ENFORCE_EQ(lengthsB.dim32(0), lengthsA.dim32(0));

    int N = lengthsB.dim32(0);

    auto* A_data = A.template data<T>();
    auto* B_data = B.template data<T>();
    const int* A_input_len = lengthsA.template data<int>();
    const int* B_input_len = lengthsB.template data<int>();

    auto* output_matmul_values = Output(0);
    auto* output_matmul_lengths = Output(1);

    output_matmul_lengths->Resize(N);

    int* output_matmul_lengths_data =
        output_matmul_lengths->template mutable_data<int>();

    int total_output_length = 0;
    for (int i = 0; i < N; i++) {
      total_output_length += A_input_len[i] * B_input_len[i];
      output_matmul_lengths_data[i] = A_input_len[i] * B_input_len[i];
    }
    std::vector<int> output_dims;
    if (A.ndim() == 1) {
      int orig_dim = 0;
      for (int i = 0; i < N; i++) {
        orig_dim += A_input_len[i] == 0 ? 0 : A_input_len[i] / B_input_len[i];
      }
      output_dims = std::vector<int>({orig_dim, B.dim32(1)});
      orig_dim *= B.dim32(1);
      output_matmul_values->Resize(orig_dim);
      output_matmul_values->Reshape(output_dims);
    } else {
      output_matmul_values->Resize(total_output_length);
    }
    auto* output_matmul_values_data =
        output_matmul_values->template mutable_data<T>();

    auto* cur_A_data = A_data;
    auto* cur_B_data = B_data;
    int out_dist = 0;
    for (int i = 0; i < N; ++i) {
      if (A_input_len[i] == 0 || B_input_len[i] == 0) {
        continue;
      }

      int a_dim0 =
          (A.ndim() == 1) ? A_input_len[i] / B_input_len[i] : A_input_len[i];
      int a_dim1 = (A.ndim() == 1) ? B_input_len[i] : A.dim32(1);
      int b_dim0 = B_input_len[i];
      int b_dim1 = B.dim32(1);
      int swap = 0;
      if (trans_a_) {
        swap = a_dim0;
        a_dim0 = a_dim1;
        a_dim1 = swap;
      }
      if (trans_b_) {
        swap = b_dim0;
        b_dim0 = b_dim1;
        b_dim1 = swap;
      }
      math::Gemm<T, Context, Engine>(
          trans_a_ ? CblasTrans : CblasNoTrans,
          trans_b_ ? CblasTrans : CblasNoTrans,
          a_dim0,
          b_dim1,
          a_dim1,
          1,
          cur_A_data,
          cur_B_data,
          0,
          output_matmul_values_data + out_dist,
          &context_);
      out_dist += a_dim0 * b_dim1;
      cur_A_data += a_dim1 * a_dim0;
      cur_B_data += b_dim1 * b_dim0;
    }
    return true;
  }

 protected:
  bool trans_a_;
  bool trans_b_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LENGTHS_MATMUL_OP_H_
