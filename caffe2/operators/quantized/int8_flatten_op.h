#ifndef CAFFE2_OPERATORS_INT8_FLATTEN_OP_H_
#define CAFFE2_OPERATORS_INT8_FLATTEN_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

class Int8FlattenOp : public Operator<CPUContext> {
 public:
  Int8FlattenOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        axis_(this->template GetSingleArgument<int>("axis", 1)) {}

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->GetMutable<Int8TensorCPU>();
    int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    CHECK_EQ(Y_offset, X.zero_point);
    CHECK_EQ(Y_scale, X.scale);
    Y->scale = Y_scale;
    Y->zero_point = Y_offset;
    CAFFE_ENFORCE_GE(
        X.t.sizes().size(), axis_, "The rank of the tensor must be >= axis.");
    Y->t.Resize(X.t.size_to_dim(axis_), X.t.size_from_dim(axis_));
    context_.CopyItemsToCPU(
        X.t.dtype(),
        X.t.numel(),
        X.t.raw_data(),
        Y->t.raw_mutable_data(X.t.dtype()));
    return true;
  }

 private:
  int axis_;
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_FLATTEN_OP_H_
