#ifndef CAFFE2_OPERATORS_SCALE_BLOBS_OP_H_
#define CAFFE2_OPERATORS_SCALE_BLOBS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "c10/util/irange.h"

namespace caffe2 {

template <class Context>
class ScaleBlobsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit ScaleBlobsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(float, "scale", scale_, 1.0f) {}

  template <typename T>
  bool DoRunWithType() {
    int batchSize = InputSize();

    for (const auto i : c10::irange(batchSize)) {
      const auto& X = Input(i);
      auto* Y = Output(i, X.sizes(), at::dtype<T>());
      math::Scale<float, T, Context>(
          X.numel(),
          scale_,
          X.template data<T>(),
          Y->template mutable_data<T>(),
          &context_);
    }
    return true;
  }

  bool RunOnDevice() override {
    for (const auto i : c10::irange(InputSize())) {
      auto& input = this->template Input<Tensor>(i, CPU);
      auto* output = this->template Output<Tensor>(i, CPU);
      output->ResizeLike(input);
    }
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

 private:
  const float scale_;
  Tensor blobSizes_;
  Tensor inputs_;
  Tensor outputs_;

  Tensor hostBlobSizes_;
  Tensor hostInputs_;
  Tensor hostOutputs_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SCALE_BLOBS_OP_H_
