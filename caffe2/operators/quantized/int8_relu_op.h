#ifndef CAFFE2_OPERATORS_INT8_RELU_OP_H_
#define CAFFE2_OPERATORS_INT8_RELU_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

namespace int8 {

class Int8ReluOp final : public Operator<CPUContext> {
 public:
  using Operator<CPUContext>::Operator;

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    Y->t.ResizeLike(X.t);
    Y->scale = X.scale;
    Y->zero_point = X.zero_point;
    CHECK_GE(X.zero_point, std::numeric_limits<uint8_t>::min());
    CHECK_LE(X.zero_point, std::numeric_limits<uint8_t>::max());
    const int32_t Y_offset =
        this->template GetSingleArgument<int>("Y_zero_point", 0);
    const float Y_scale =
        this->template GetSingleArgument<float>("Y_scale", 1.0f);
    CHECK_EQ(Y_offset, X.zero_point);
    CHECK_EQ(Y_scale, X.scale);
    EigenVectorMap<uint8_t>(Y->t.mutable_data<uint8_t>(), X.t.numel()) =
        ConstEigenVectorMap<uint8_t>(X.t.data<uint8_t>(), X.t.numel())
            .cwiseMax(QuantizeUint8(X.scale, X.zero_point, 0));
    return true;
  }
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_RELU_OP_H_
