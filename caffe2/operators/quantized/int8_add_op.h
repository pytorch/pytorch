#ifndef CAFFE2_OPERATORS_INT8_ADD_OP_H_
#define CAFFE2_OPERATORS_INT8_ADD_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_simd.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

namespace {

/*
 * Implementation based on TensorFlow Lite kernels:
 * - Repo: https://github.com/tensorflow/tensorflow
 * - Path: tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h
 * - Hash: d4ad9c73969c45d1a224ebfc43eb645b9860216b
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

constexpr size_t kAddLeftShift = 20;

void Int8Add(
    const uint8_t* X0_data,
    size_t N,
    size_t D,
    int32_t X0_offset,
    int32_t X0_multiplier,
    int X0_shift,
    const uint8_t* X1_data,
    int32_t X1_offset,
    int32_t X1_multiplier,
    int X1_shift,
    int32_t Y_offset,
    int32_t Y_multiplier,
    int Y_shift,
    uint8_t* Y_data,
    uint8_t Y_activation_min,
    uint8_t Y_activation_max,
    ThreadPool* threadPool) {
  CHECK_GT(X0_offset, -256);
  CHECK_GT(X1_offset, -256);
  CHECK_LT(X0_offset, 256);
  CHECK_LT(X1_offset, 256);
  static_assert(kAddLeftShift > 5, "");
  static_assert(kAddLeftShift <= 20, "");

  auto f = [&](int, size_t n) {
    size_t d = 0;

#ifdef INT8_NEON_SIMD
    constexpr size_t kIntermediateAddLeftShift = 4;
    const auto X0_offset_val =
        vshlq_n_s16(vdupq_n_s16(X0_offset), kIntermediateAddLeftShift);
    const auto X1_offset_val =
        vshlq_n_s16(vdupq_n_s16(X1_offset), kIntermediateAddLeftShift);
    const auto X0_shift_dup = vdupq_n_s32(-X0_shift);
    const auto X1_shift_dup = vdupq_n_s32(-X1_shift);
    const auto DUnroll = (D / 8) * 8;

    for (; d < DUnroll; d += 8) {
      const auto X0_val_original = vld1_u8(X0_data + n * D + d);
      const auto X1_val_original = vld1_u8(X1_data + n * D + d);

      // Load input
      // Widen to int16
      // Add int16 offset.
      // Widen to int32
      // Shift right by 20.
      // Alternatively, we can widening shift X by 4, shifty X0_offset by 4,
      // add, then shift by 16. Safe as X << 5 + X_offset << 5 can't overflow
      // uint16, as X ~ 8 bit, X_offset ~ 10 bit, so 15 bits total from X +
      // X_offset
      const auto X0_val_s16 = vreinterpretq_s16_u16(
          vshll_n_u8(X0_val_original, kIntermediateAddLeftShift));
      const auto X1_val_s16 = vreinterpretq_s16_u16(
          vshll_n_u8(X1_val_original, kIntermediateAddLeftShift));
      const auto X0_val = vaddq_s16(X0_val_s16, X0_offset_val);
      const auto X1_val = vaddq_s16(X1_val_s16, X1_offset_val);
      const auto X0_val_high = vget_high_s16(X0_val);
      const auto X0_val_low = vget_low_s16(X0_val);
      const auto X1_val_high = vget_high_s16(X1_val);
      const auto X1_val_low = vget_low_s16(X1_val);
      auto x11 =
          vshll_n_s16(X0_val_low, kAddLeftShift - kIntermediateAddLeftShift);
      auto x12 =
          vshll_n_s16(X0_val_high, kAddLeftShift - kIntermediateAddLeftShift);
      auto x21 =
          vshll_n_s16(X1_val_low, kAddLeftShift - kIntermediateAddLeftShift);
      auto x22 =
          vshll_n_s16(X1_val_high, kAddLeftShift - kIntermediateAddLeftShift);
      x11 = vqrdmulhq_n_s32(x11, X0_multiplier);
      x12 = vqrdmulhq_n_s32(x12, X0_multiplier);
      x21 = vqrdmulhq_n_s32(x21, X1_multiplier);
      x22 = vqrdmulhq_n_s32(x22, X1_multiplier);
      x11 = vshlq_s32(x11, X0_shift_dup);
      x12 = vshlq_s32(x12, X0_shift_dup);
      x21 = vshlq_s32(x21, X1_shift_dup);
      x22 = vshlq_s32(x22, X1_shift_dup);
      auto s1 = vaddq_s32(x11, x21);
      auto s2 = vaddq_s32(x12, x22);
      s1 = vqrdmulhq_n_s32(s1, Y_multiplier);
      s2 = vqrdmulhq_n_s32(s2, Y_multiplier);
      using gemmlowp::RoundingDivideByPOT;
      s1 = RoundingDivideByPOT(s1, Y_shift);
      s2 = RoundingDivideByPOT(s2, Y_shift);
      const auto s1_narrowed = vmovn_s32(s1);
      const auto s2_narrowed = vmovn_s32(s2);
      const auto s = vaddq_s16(
          vcombine_s16(s1_narrowed, s2_narrowed), vdupq_n_s16(Y_offset));
      auto ss = vqmovun_s16(s);
      ss = vmin_u8(ss, vdup_n_u8(Y_activation_max));
      ss = vmax_u8(ss, vdup_n_u8(Y_activation_min));
      vst1_u8(Y_data + n * D + d, ss);
    }
#endif // NEON

    for (; d < D; d++) {
      const int32_t X0_val = X0_offset + X0_data[n * D + d];
      const int32_t X1_val = X1_offset + X1_data[n * D + d];
      const int32_t shifted_X0_val = X0_val * (1 << kAddLeftShift);
      const int32_t shifted_X1_val = X1_val * (1 << kAddLeftShift);
      const int32_t scaled_X0_val = MultiplyByQuantizedMultiplierSmallerThanOne(
          shifted_X0_val, X0_multiplier, X0_shift);
      const int32_t scaled_X1_val = MultiplyByQuantizedMultiplierSmallerThanOne(
          shifted_X1_val, X1_multiplier, X1_shift);
      const int32_t raw_sum = scaled_X0_val + scaled_X1_val;
      const int32_t raw_Y = MultiplyByQuantizedMultiplierSmallerThanOne(
                                raw_sum, Y_multiplier, Y_shift) +
          Y_offset;
      const int32_t clamped_Y = std::min<int32_t>(
          Y_activation_max, std::max<int32_t>(Y_activation_min, raw_Y));
      Y_data[n * D + d] = static_cast<uint8_t>(clamped_Y);
    }
  };
  threadPool->run(f, N);
}

} // namespace

template <Activation Ac>
class Int8AddOp final : public Operator<CPUContext> {
 public:
  Int8AddOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        ws_(ws) {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE_EQ(Inputs().size(), 2);
    const auto& X0 = Inputs()[0]->template Get<Int8TensorCPU>();
    const auto& X1 = Inputs()[1]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    auto X0_offset = -X0.zero_point;
    auto X1_offset = -X1.zero_point;
    int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    const double twice_max_input_scale = 2 * std::max(X0.scale, X1.scale);
    const double real_X0_multiplier = X0.scale / twice_max_input_scale;
    const double real_X1_multiplier = X1.scale / twice_max_input_scale;
    const double real_Y_multiplier =
        twice_max_input_scale / ((1 << kAddLeftShift) * Y_scale);

    Y->t.ResizeLike(X0.t);
    Y->zero_point = Y_offset;
    Y->scale = Y_scale;

    int32_t X0_multiplier;
    int X0_shift;
    QuantizeMultiplierSmallerThanOne(
        real_X0_multiplier, &X0_multiplier, &X0_shift);
    int32_t X1_multiplier;
    int X1_shift;
    QuantizeMultiplierSmallerThanOne(
        real_X1_multiplier, &X1_multiplier, &X1_shift);
    int32_t Y_multiplier;
    int Y_shift;
    QuantizeMultiplierSmallerThanOne(
        real_Y_multiplier, &Y_multiplier, &Y_shift);

    Int8Add(
        X0.t.template data<uint8_t>(),
        X0.t.numel() / X0.t.size(X0.t.dim() - 1),
        X0.t.size(X0.t.dim() - 1),
        X0_offset,
        X0_multiplier,
        X0_shift,
        X1.t.template data<uint8_t>(),
        X1_offset,
        X1_multiplier,
        X1_shift,
        Y_offset,
        Y_multiplier,
        Y_shift,
        Y->t.template mutable_data<uint8_t>(),
        activationLimits(Y->scale, Y->zero_point, Ac).first,
        activationLimits(Y->scale, Y->zero_point, Ac).second,
        ws_->GetThreadPool());
    return true;
  }

 private:
  Workspace* ws_;
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_ADD_OP_H_
