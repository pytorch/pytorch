#ifndef CAFFE2_OPERATORS_FLATTEN_OP_H_
#define CAFFE2_OPERATORS_FLATTEN_OP_H_

#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class FlattenOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  FlattenOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(this->template GetSingleArgument<int>("axis", 1)) {}

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    CAFFE_ENFORCE_GE(
        input.sizes().size(), axis_, "The rank of the tensor must be >= axis.");
    output->Resize(input.size_to_dim(axis_), input.size_from_dim(axis_));
    context_.CopyItemsSameDevice(
        input.dtype(),
        input.numel(),
        input.raw_data(),
        output->raw_mutable_data(input.dtype()));
    return true;
  }

 private:
  int axis_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FLATTEN_OP_H_
