#ifndef CAFFE2_OPERATORS_INT8_QUANTIZE_OP_H_
#define CAFFE2_OPERATORS_INT8_QUANTIZE_OP_H_

#include <fbgemm/FbgemmConvert.h>
#include <cmath>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"
#include "fp16_fma.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);

namespace caffe2 {

namespace int8 {

namespace {


static float ClampScale(float s)
{
  const float MinScale(1e-10f);
    if (std::fabs(s) < MinScale) {
        LOG_EVERY_N(WARNING, 1000) << "Too small scale detected: "
            << s << " clamping to +/-" << MinScale;
        return std::signbit(s) ? -MinScale : MinScale;
    } else {
        return s;
    }
}

void Int8QuantizeNNPI(
    const float* in,
    uint8_t* out,
    const int64_t N,
    const float Y_scale,
    const int32_t Y_offset) {
  const int32_t qmin = std::numeric_limits<uint8_t>::min();
  const int32_t qmax = std::numeric_limits<uint8_t>::max();

  float inv_scale = ClampScale(1 / Y_scale);
  float inv_scale_fp16 = 0;
  fbgemm::RoundToFloat16(
      &inv_scale, &inv_scale_fp16, 1, false /* no clamping */);
  float offset_tmp = -Y_offset;
  fbgemm::RoundToFloat16(
      &offset_tmp, &offset_tmp, 1, false /* no clamping */);
  std::vector<float> in_fp16(N);
  fbgemm::RoundToFloat16(
      in, in_fp16.data(), N, false /* no clamping */);

  std::vector<float> inv_scalev(N, inv_scale_fp16);
  std::vector<float> offsetv(N, -offset_tmp);
  fake_fp16::fma_fp16(N, in_fp16.data(), inv_scalev.data(), offsetv.data());
  for (int i = 0; i < N; i++) {
    offsetv[i] = round(offsetv[i]);
  }
  fbgemm::RoundToFloat16(
      offsetv.data(), offsetv.data(), N, false /* no clamping */);
  for (int i = 0; i < N; i++) {
    float halfRes = offsetv[i];
    if (std::isinf(halfRes)) {
      if (halfRes > 0) {
        halfRes = qmax;
      } else {
        halfRes = qmin;
      }
    }
    if (halfRes > qmax) {
      halfRes = qmax;
    }
    if (halfRes < qmin) {
      halfRes = qmin;
    }
    out[i] = static_cast<uint8_t>(halfRes);
  }
}

} // namespace

class Int8QuantizeNNPIOp final : public Operator<CPUContext> {
 public:
  using Operator<CPUContext>::Operator;

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    Y->t.ResizeLike(X);
    int32_t Y_offset =
        this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    Y->scale = Y_scale;
    Y->zero_point = Y_offset;
    Int8QuantizeNNPI(
        X.data<float>(),
        Y->t.mutable_data<uint8_t>(),
        X.numel(),
        Y_scale,
        Y_offset);
    return true;
  }
};

} // namespace int8
} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_QUANTIZE_OP_H_
