#ifndef CAFFE2_OPERATORS_SPARSE_MATRIX_RESHAPE_H_
#define CAFFE2_OPERATORS_SPARSE_MATRIX_RESHAPE_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class SparseMatrixReshapeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseMatrixReshapeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    CAFFE_ENFORCE(
        OperatorBase::HasArgument("old_shape"),
        "Argument `old_shape` is missing.");
    CAFFE_ENFORCE(
        OperatorBase::HasArgument("new_shape"),
        "Argument `new_shape` is missing.");

    vector<TIndex> old_shape =
        OperatorBase::GetRepeatedArgument<TIndex>("old_shape");
    vector<TIndex> new_shape =
        OperatorBase::GetRepeatedArgument<TIndex>("new_shape");

    CAFFE_ENFORCE(
        old_shape.size() == 2,
        "Argument `old_shape` must contain exactly two integers.");
    CAFFE_ENFORCE(
        new_shape.size() == 2,
        "Argument `new_shape` must contain exactly two integers.");

    CAFFE_ENFORCE(
        old_shape[1] > 0,
        "The second dimension in argument `old_shape` must be positive.");

    old_stride_ = old_shape[1];

    if (old_shape[0] == -1) {
      CAFFE_ENFORCE(
          new_shape[1] > 0,
          "The second dimension in `new_shape` must be positive.");
    } else {
      CAFFE_ENFORCE(
          old_shape[0] > 0,
          "The first dimension in `old_shape` must be positive.");

      TIndex matrix_size = old_shape[0] * old_shape[1];

      if (new_shape[0] == -1) {
        CAFFE_ENFORCE(
            new_shape[1] > 0,
            "Only one dimension in argument `new_shape` can be -1.");
        CAFFE_ENFORCE(
            matrix_size % new_shape[1] == 0,
            "Argument `new_shape` does not agree with `old_shape`.");
      } else {
        CAFFE_ENFORCE(
            new_shape[0] > 0 && (new_shape[1] == -1 || new_shape[1] > 0),
            "Dimensions in argument `new_shape` must be positive or -1.");
        if (new_shape[1] == -1) {
          CAFFE_ENFORCE(
              matrix_size % new_shape[0] == 0,
              "Argument `new_shape` does not agree with `old_shape`.");
          new_shape[1] = matrix_size / new_shape[0];
        } else {
          CAFFE_ENFORCE(
              new_shape[0] * new_shape[1] == matrix_size,
              "Argument `new_shape` does not agree with `old_shape`.");
        }
      }
    }
    new_stride_ = new_shape[1];
  }

  bool RunOnDevice() override {
    auto& old_col = Input(0);
    CAFFE_ENFORCE(old_col.ndim() == 1, "Row index tensor must be 1-D.");
    auto& old_row = Input(1);
    CAFFE_ENFORCE(old_row.ndim() == 1, "Column index tensor must be 1-D.");

    const auto nnz = old_col.size();
    CAFFE_ENFORCE(
        old_row.size() == nnz,
        "Column and row tensors must have the same size.");

    auto* new_col = Output(0);
    auto* new_row = Output(1);
    new_col->Resize(nnz);
    new_row->Resize(nnz);

    const auto* old_col_data = old_col.template data<TIndex>();
    const auto* old_row_data = old_row.template data<int>();

    auto* new_col_data = new_col->template mutable_data<TIndex>();
    auto* new_row_data = new_row->template mutable_data<int>();

    for (int i = 0; i < nnz; ++i) {
      TIndex offset = old_row_data[i] * old_stride_ + old_col_data[i];
      new_row_data[i] = offset / new_stride_;
      new_col_data[i] = offset % new_stride_;
    }

    return true;
  }

 private:
  TIndex old_stride_;
  TIndex new_stride_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SPARSE_MATRIX_RESHAPE_H_
