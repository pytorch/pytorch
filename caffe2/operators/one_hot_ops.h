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
  bool DoRunWithType();

 protected:
  INPUT_TAGS(X, LENS, VALS);
  OUTPUT_TAGS(ONE_HOT);

 private:
  // allows for fast random access to a given dict and is re-used across runs
  std::vector<TIndex> valsOffsets_;
};

template <class Context>
class BatchBucketOneHotOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BatchBucketOneHotOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X, LENS, BOUNDARIES);
  OUTPUT_TAGS(ONE_HOT);
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_ONE_HOT_OPS_H_
