#ifndef CAFFE2_OPERATORS_INT8_SWISH_OP_H_
#define CAFFE2_OPERATORS_INT8_SWISH_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

namespace {
using namespace std;
void SwishFakeInt8NNPI(
    const uint8_t* in,
    uint8_t* out,
    const int64_t N,
    const float X_scale,
    const int32_t X_offset,
    const float Y_scale,
    const int32_t Y_offset) {

  const uint8_t max_val = std::numeric_limits<uint8_t>::max();
  const uint8_t min_val = std::numeric_limits<uint8_t>::min();
  float X_scale_fp32 = 1.0f / X_scale;
  float deq_val = 0.0f;
  float deq_swish = 0.0f;
  int32_t quant_val = 0;
  uint8_t result = 0;

  for (const auto i : c10::irange(N)) {
    deq_val = (static_cast<uint8_t>(in[i]) - X_offset) / X_scale_fp32;
    deq_swish = deq_val / (1 + exp(-deq_val));
    quant_val = round(deq_swish / Y_scale + Y_offset);
    result = quant_val;
    if (quant_val > max_val) {
      result = max_val;
    }
    if (quant_val < min_val) {
      result = min_val;
    }
    out[i] = static_cast<uint8_t>(result);
  }
}

} // namespace


class SwishInt8NNPIOp final : public Operator<CPUContext> {
 public:
  using Operator<CPUContext>::Operator;

  template <class... Args>
  explicit SwishInt8NNPIOp(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    Y->t.ResizeLike(X.t);

    int32_t Y_offset_ =
      this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale_ = this->template GetSingleArgument<float>("Y_scale", 1);

    Y->scale = Y_scale_;
    Y->zero_point = Y_offset_;

    SwishFakeInt8NNPI(
        X.t.data<uint8_t>(),
        Y->t.mutable_data<uint8_t>(),
        X.t.numel(),
        X.scale,
        X.zero_point,
        Y_scale_,
        Y_offset_);
    return true;
  }

};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_SWISH_OP_H_
