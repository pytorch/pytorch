#ifndef CAFFE2_OPERATORS_INT8_DEQUANTIZE_OP_H_
#define CAFFE2_OPERATORS_INT8_DEQUANTIZE_OP_H_

#include <fbgemm/FbgemmConvert.h>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);

namespace caffe2 {

namespace int8 {

namespace {

void Int8DequantizeNNPI(
    const uint8_t* in,
    float* out,
    const int64_t N,
    const float X_scale,
    const int32_t X_offset) {
  float X_scale_fp32 = 1.0f / X_scale;
  for (const auto i : c10::irange(N)) {
    out[i] = (float)(static_cast<int32_t>(in[i]) - X_offset) / X_scale_fp32;
  }
} // namespace

} // namespace

class Int8DequantizeNNPIOp final : public Operator<CPUContext> {
 public:
  using Operator<CPUContext>::Operator;

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();

    auto* Y = Output(0, X.t.sizes(), at::dtype<float>());
    int32_t X_offset = X.zero_point;
    auto X_scale = X.scale;
    Int8DequantizeNNPI(
        X.t.data<uint8_t>(),
        Y->mutable_data<float>(),
        X.t.numel(),
        X_scale,
        X_offset);
    // UsingOneOverScale_);
    return true;
  }
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_DEQUANTIZE_OP_H_
