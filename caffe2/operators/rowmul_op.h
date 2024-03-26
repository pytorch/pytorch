#ifndef CAFFE2_OPERATORS_ROW_MUL_H_
#define CAFFE2_OPERATORS_ROW_MUL_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "c10/util/irange.h"

namespace caffe2 {

// A hacky version of Mul with broadcast
// RowMul([mat, w], [output])
template <typename T, class Context>
class RowMulOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(RowMulOp);

  bool RunOnDevice() override {
    auto& mat = Input(0);
    auto& w = Input(1);

    auto* output = Output(0, mat.sizes(), at::dtype<T>());
    T* output_data = output->template mutable_data<T>();
    const T* mat_data = mat.template data<T>();
    const T* w_data = w.template data<T>();

    // Dimension checking
    CAFFE_ENFORCE_EQ(
        w.numel(),
        mat.dim32(0),
        "Length of w should be equal to the first dim of mat");

    auto block_size = mat.size_from_dim(1);
    for (const auto i : c10::irange(w.numel())) {
      size_t offset = i * block_size;
      for (const auto j : c10::irange(block_size)) {
        output_data[offset + j] = mat_data[offset + j] * w_data[i];
      }
    }

    return true;
  }
};

// A hacky version
template <typename T, class Context>
class ReduceTailSumOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(ReduceTailSumOp);

  bool RunOnDevice() override {
    auto& mat = Input(0);

    int N = mat.dim32(0);
    int block_size = mat.size_from_dim(1);

    auto* output = Output(0, {N}, at::dtype<T>());
    T* output_data = output->template mutable_data<T>();
    const T* mat_data = mat.template data<T>();

    for (const auto i : c10::irange(N)) {
      output_data[i] = 0;
      size_t offset = i * block_size;
      for (const auto j : c10::irange(block_size)) {
        output_data[i] += mat_data[offset + j];
      }
    }
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ROW_MUL_H_
