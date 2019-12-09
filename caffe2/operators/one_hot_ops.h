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

  template <class... Args>
  explicit OneHotOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    auto& indices = Input(0);
    CAFFE_ENFORCE_EQ(
        indices.dim(),
        1,
        "indices input must be 1D tensor of data type int64_t");

    // Index size input must be in CPU context
    auto& index_size_tensor = this->template Input<Tensor>(1, CPU);
    CAFFE_ENFORCE_EQ(
        index_size_tensor.numel(),
        1,
        "index_size_tensor input must be scalar of data type int64_t");

    auto batch_size = indices.numel();
    auto index_size = *index_size_tensor.template data<int64_t>();
    auto one_hots = Output(0);
    one_hots->Resize(batch_size, index_size);
    auto output_size = one_hots->numel();
    if (output_size == 0) {
      return true;
    }

    DoOneHotOp(batch_size, index_size, indices, one_hots);
    return true;
  }

 protected:
  void DoOneHotOp(
      int64_t batch_size,
      int64_t index_size,
      const Tensor& indices,
      Tensor* output);
};

template <class Context>
class BatchOneHotOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit BatchOneHotOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(X));
  }

  template <typename T>
  bool DoRunWithType();

  INPUT_TAGS(X, LENS, VALS);

 protected:
  OUTPUT_TAGS(ONE_HOT);

 private:
  // allows for fast random access to a given dict and is re-used across runs
  std::vector<int64_t> valsOffsets_;
};

template <class Context>
class BatchBucketOneHotOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit BatchBucketOneHotOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X, LENS, BOUNDARIES);
  OUTPUT_TAGS(ONE_HOT);
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_ONE_HOT_OPS_H_
