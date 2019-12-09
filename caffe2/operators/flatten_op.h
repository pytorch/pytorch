#ifndef CAFFE2_OPERATORS_FLATTEN_OP_H_
#define CAFFE2_OPERATORS_FLATTEN_OP_H_

#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class FlattenOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit FlattenOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        axis_(this->template GetSingleArgument<int>("axis", 1)) {}

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    CAFFE_ENFORCE_GE(
        input.dim(), axis_, "The rank of the tensor must be >= axis.");
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

inline std::vector<TensorShape> TensorInferenceForFlatten(
    const OperatorDef& def,
    const std::vector<TensorShape>& in) {
  ArgumentHelper helper(def);
  const int axis = helper.GetSingleArgument<int>("axis", 1);
  std::vector<TensorShape> out(1);
  int64_t outer = 1;
  int64_t inner = 1;
  std::size_t index = 0;
  for (auto d : in[0].dims()) {
    if (index < axis) {
      outer *= d;
    } else {
      inner *= d;
    }
    ++index;
  }
  out[0].set_data_type(in[0].data_type());
  out[0].add_dims(outer);
  out[0].add_dims(inner);
  return out;
}

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FLATTEN_OP_H_
