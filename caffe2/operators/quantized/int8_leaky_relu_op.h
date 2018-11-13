#ifndef CAFFE2_OPERATORS_INT8_LEAKY_RELU_OP_H_
#define CAFFE2_OPERATORS_INT8_LEAKY_RELU_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

class Int8LeakyReluOp final : public Operator<CPUContext> {
 public:
  Int8LeakyReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    double alpha = this->template GetSingleArgument<float>("alpha", 0.01);
    CAFFE_ENFORCE_GT(alpha, 0.0);
    CAFFE_ENFORCE_LT(alpha, 1.0);
    QuantizeMultiplierSmallerThanOne(alpha, &multiplier_, &shift_);
  }

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    Y->t.ResizeLike(X.t);
    Y->scale = X.scale;
    Y->zero_point = X.zero_point;
    CHECK_GE(X.zero_point, std::numeric_limits<uint8_t>::min());
    CHECK_LE(X.zero_point, std::numeric_limits<uint8_t>::max());
    int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    CHECK_EQ(Y_offset, X.zero_point);
    CHECK_EQ(Y_scale, X.scale);

    const uint8_t* Xdata = X.t.data<uint8_t>();
    uint8_t* Ydata = Y->t.mutable_data<uint8_t>();

    // For x < zero_point:
    //   (y - zero_point) * scale = alpha * (x - zero_point) * scale
    //   y = alpha * (x - zeropoint) + zero_point
    for (int i = 0; i < X.t.numel(); i++) {
      if (Xdata[i] < X.zero_point) {
        int32_t out = MultiplyByQuantizedMultiplierSmallerThanOne(
                          Xdata[i] - X.zero_point, multiplier_, shift_) +
            X.zero_point;
        Ydata[i] = static_cast<uint8_t>(out);
      } else {
        Ydata[i] = Xdata[i];
      }
    }
    return true;
  }

 private:
  int32_t multiplier_;
  int shift_;
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_LEAKY_RELU_OP_H_
