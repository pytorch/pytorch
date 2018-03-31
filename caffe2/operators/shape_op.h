
#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// RecordShapeOp records the shape of the input tensor to a vector of int. You
// mostly don't need this operator explicitly, and it is mostly used in the
// autodiff process.
template <class Context>
class ShapeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(ShapeOp);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = OperatorBase::Output<Tensor<Context>>(0);
    output->Resize(input.ndim());
    TIndex* output_data = output->template mutable_data<TIndex>();
    context_.template CopyBytes<Context, Context>(
        input.ndim() * sizeof(TIndex), input.dims().data(), output_data);
    return true;
  }
};

} // namespace caffe2
