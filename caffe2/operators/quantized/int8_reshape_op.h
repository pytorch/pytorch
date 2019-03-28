#ifndef CAFFE2_OPERATORS_INT8_RESHAPE_OP_H_
#define CAFFE2_OPERATORS_INT8_RESHAPE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"
#include "caffe2/operators/reshape_op.h"

namespace caffe2 {

namespace int8 {

class Int8ReshapeOp final : public ReshapeOp<uint8_t, CPUContext> {
 public:
  template <class... Args>
  explicit Int8ReshapeOp(Args&&... args)
      : ReshapeOp(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    if (InputSize() == 2) {
      return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(1));
    }
    CAFFE_ENFORCE(
        OperatorBase::HasArgument("shape"), "Argument `shape` is missing.");
    return this->template DoRunWithType<int64_t>();
  }

  template <typename T>
  bool DoRunWithType() {
    auto& X = Inputs()[0]->Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->GetMutable<Int8TensorCPU>();
    int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    CHECK_EQ(Y_offset, X.zero_point);
    CHECK_EQ(Y_scale, X.scale);
    Y->scale = Y_scale;
    Y->zero_point = Y_offset;
    DoRunWithTypeImpl<T>(X.t, &Y->t);
    return true;
  }
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_RESHAPE_OP_H_
