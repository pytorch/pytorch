#ifndef CAFFE_OPERATORS_ONE_HOT_OPS_H_
#define CAFFE_OPERATORS_ONE_HOT_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class OneHotOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  OneHotOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& indices = Input(0);
    CAFFE_ENFORCE_EQ(
        indices.ndim(),
        1,
        "indices input must be 1D tensor of data type TIndex");

    // Index size input must be in CPU context
    auto& index_size_tensor = OperatorBase::Input<Tensor<CPUContext>>(1);
    CAFFE_ENFORCE_EQ(
        index_size_tensor.size(),
        1,
        "index_size_tensor input must be scalar of data type TIndex");

    auto batch_size = indices.size();
    auto index_size = *index_size_tensor.template data<TIndex>();
    auto one_hots = Output(0);
    one_hots->Resize(batch_size, index_size);
    auto output_size = one_hots->size();
    if (output_size == 0) {
      return true;
    }

    DoOneHotOp(batch_size, index_size, indices, one_hots);
    return true;
  }

 protected:
  void DoOneHotOp(
      TIndex batch_size,
      TIndex index_size,
      const Tensor<Context>& indices,
      Tensor<Context>* output);
};

template <class Context>
class BatchOneHotOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BatchOneHotOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(X));
  }

  template <typename T>
  bool DoRunWithType() {
    auto& input = Input(X);
    auto& lens = Input(LENS);
    auto& vals = Input(VALS);
    CAFFE_ENFORCE_GE(input.ndim(), 2);
    auto batch_size = input.dim(0);
    auto in_dim = input.size_from_dim(1);
    CAFFE_ENFORCE_GE(in_dim, 1);
    CAFFE_ENFORCE_GE(batch_size, 1);
    CAFFE_ENFORCE_EQ(lens.size(), in_dim);

    auto output = Output(ONE_HOT);
    output->Resize(batch_size, vals.size());
    if (output->size() == 0) {
      return true;
    }
    return DoBatchOneHotOp<T>(batch_size, in_dim, input, lens, vals, output);
  }

 protected:
  template <typename T>
  bool DoBatchOneHotOp(
      TIndex batch_size,
      TIndex in_dim,
      const Tensor<Context>& input,
      const Tensor<Context>& lens,
      const Tensor<Context>& vals,
      Tensor<Context>* output);

  INPUT_TAGS(X, LENS, VALS);
  OUTPUT_TAGS(ONE_HOT);
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_ONE_HOT_OPS_H_
