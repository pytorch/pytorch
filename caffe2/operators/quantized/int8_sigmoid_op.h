#ifndef CAFFE2_INT8_SIGMOID_OP_H_
#define CAFFE2_INT8_SIGMOID_OP_H_
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

int SigmoidPrepare(
    double input_scale,
    int32_t* quantized_multiplier,
    int32_t* left_shift) {
  static constexpr int kInputIntegerBits = 4;

  const double input_real_multiplier =
      input_scale * static_cast<double>(1 << (31 - kInputIntegerBits));

  QuantizeMultiplierGreaterThanOne(
      input_real_multiplier, quantized_multiplier, left_shift);
  return CalculateInputRadius(kInputIntegerBits, *left_shift);
}

inline void Int8Logistic(
    const uint8_t* input_data,
    uint8_t* output_data,
    const int32_t input_zero_point,
    const int32_t input_range_radius,
    const int32_t input_multiplier,
    const int32_t input_left_shift,
    const int32_t size) {
  int c = 0;
#ifdef INT8_NEON_SIMD
  // Handle 16 values at a time
  for (; c <= size - 16; c += 16) {
    // Read input uint8_t values, cast to int16 and subtract input_zero_point
    uint8x16_t input_val_u8 = vld1q_u8(input_data + c);
    int16x8_t input_val_centered_0 = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_val_u8))),
        vdupq_n_s16(input_zero_point));
    int16x8_t input_val_centered_1 = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(input_val_u8))),
        vdupq_n_s16(input_zero_point));

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = 0;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 255;
    //   } else {
    //     ...
    uint16x8_t mask_rightclamp_0 =
        vcgtq_s16(input_val_centered_0, vdupq_n_s16(input_range_radius));
    uint16x8_t mask_rightclamp_1 =
        vcgtq_s16(input_val_centered_1, vdupq_n_s16(input_range_radius));
    uint16x8_t mask_leftclamp_0 =
        vcgeq_s16(input_val_centered_0, vdupq_n_s16(-input_range_radius));
    uint16x8_t mask_leftclamp_1 =
        vcgeq_s16(input_val_centered_1, vdupq_n_s16(-input_range_radius));
    uint8x16_t mask_rightclamp = vcombine_u8(
        vshrn_n_u16(mask_rightclamp_0, 8), vshrn_n_u16(mask_rightclamp_1, 8));
    uint8x16_t mask_leftclamp = vcombine_u8(
        vshrn_n_u16(mask_leftclamp_0, 8), vshrn_n_u16(mask_leftclamp_1, 8));

    // This performs what is expressed in the scalar code as
    // const int32_t input_val_rescaled =
    //     MultiplyByQuantizedMultiplierGreaterThanOne(
    //         input_val_centered, input_multiplier, input_left_shift);
    int32x4_t input_val_rescaled_0 = vshlq_s32(
        vmovl_s16(vget_low_s16(input_val_centered_0)),
        vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_1 = vshlq_s32(
        vmovl_s16(vget_high_s16(input_val_centered_0)),
        vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_2 = vshlq_s32(
        vmovl_s16(vget_low_s16(input_val_centered_1)),
        vdupq_n_s32(input_left_shift));
    int32x4_t input_val_rescaled_3 = vshlq_s32(
        vmovl_s16(vget_high_s16(input_val_centered_1)),
        vdupq_n_s32(input_left_shift));
    input_val_rescaled_0 =
        vqrdmulhq_n_s32(input_val_rescaled_0, input_multiplier);
    input_val_rescaled_1 =
        vqrdmulhq_n_s32(input_val_rescaled_1, input_multiplier);
    input_val_rescaled_2 =
        vqrdmulhq_n_s32(input_val_rescaled_2, input_multiplier);
    input_val_rescaled_3 =
        vqrdmulhq_n_s32(input_val_rescaled_3, input_multiplier);

    // Invoke gemmlowp::logistic on FixedPoint wrapping int32x4_t
    using FixedPoint4 = gemmlowp::FixedPoint<int32x4_t, 4>;
    using FixedPoint0 = gemmlowp::FixedPoint<int32x4_t, 0>;
    const FixedPoint4 input_val_f4_0 =
        FixedPoint4::FromRaw(input_val_rescaled_0);
    const FixedPoint4 input_val_f4_1 =
        FixedPoint4::FromRaw(input_val_rescaled_1);
    const FixedPoint4 input_val_f4_2 =
        FixedPoint4::FromRaw(input_val_rescaled_2);
    const FixedPoint4 input_val_f4_3 =
        FixedPoint4::FromRaw(input_val_rescaled_3);
    const FixedPoint0 output_val_f0_0 = gemmlowp::logistic(input_val_f4_0);
    const FixedPoint0 output_val_f0_1 = gemmlowp::logistic(input_val_f4_1);
    const FixedPoint0 output_val_f0_2 = gemmlowp::logistic(input_val_f4_2);
    const FixedPoint0 output_val_f0_3 = gemmlowp::logistic(input_val_f4_3);

    // Divide by 2^23 as in the scalar code
    using gemmlowp::RoundingDivideByPOT;
    int32x4_t output_val_s32_0 = RoundingDivideByPOT(output_val_f0_0.raw(), 23);
    int32x4_t output_val_s32_1 = RoundingDivideByPOT(output_val_f0_1.raw(), 23);
    int32x4_t output_val_s32_2 = RoundingDivideByPOT(output_val_f0_2.raw(), 23);
    int32x4_t output_val_s32_3 = RoundingDivideByPOT(output_val_f0_3.raw(), 23);

    // Cast output values to uint8_t, saturating
    int16x8_t output_val_s16_0 = vcombine_s16(
        vqmovn_s32(output_val_s32_0), vqmovn_s32(output_val_s32_1));
    int16x8_t output_val_s16_1 = vcombine_s16(
        vqmovn_s32(output_val_s32_2), vqmovn_s32(output_val_s32_3));
    uint8x16_t output_val_u8 = vcombine_u8(
        vqmovun_s16(output_val_s16_0), vqmovun_s16(output_val_s16_1));

    // Perform the bit-masking with the bit masks computed at the beginning,
    // see the comment there.
    output_val_u8 = vorrq_u8(output_val_u8, mask_rightclamp);
    output_val_u8 = vandq_u8(output_val_u8, mask_leftclamp);

    // Store back to memory
    vst1q_u8(output_data + c, output_val_u8);
  }
#endif
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const uint8_t input_val_u8 = input_data[c];
    const int32_t input_val_centered =
        static_cast<int32_t>(input_val_u8) - input_zero_point;
    uint8_t output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered > input_range_radius) {
      output_val = 255;
    } else {
      const int32_t input_val_rescaled =
          MultiplyByQuantizedMultiplierGreaterThanOne(
              input_val_centered, input_multiplier, input_left_shift);
      using FixedPoint4 = gemmlowp::FixedPoint<int32_t, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::logistic(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int32_t output_val_s32 = RoundingDivideByPOT(output_val_f0.raw(), 23);
      if (output_val_s32 == 256) {
        output_val_s32 = 255;
      }
      CHECK_GE(output_val_s32, 0);
      CHECK_LE(output_val_s32, 255);
      output_val = static_cast<uint8_t>(output_val_s32);
    }
    output_data[c] = output_val;
  }
}

} // namespace

class Int8SigmoidOp final : public Operator<CPUContext> {
 public:
  Int8SigmoidOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    const int32_t Y_offset =
        this->template GetSingleArgument<int>("Y_zero_point", 0);
    const float Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    CHECK_EQ(Y_offset, 0);
    CHECK_EQ(Y_scale, 1. / 256);

    Y->scale = Y_scale;
    Y->zero_point = Y_offset;
    Y->t.ResizeLike(X.t);
    int32_t input_multiplier;
    int input_left_shift;
    int input_range_radius =
        SigmoidPrepare(X.scale, &input_multiplier, &input_left_shift);
    Int8Logistic(
        X.t.data<uint8_t>(),
        Y->t.mutable_data<uint8_t>(),
        X.zero_point,
        input_range_radius,
        input_multiplier,
        input_left_shift,
        X.t.numel() / X.t.size(0));
    return true;
  }
};

} // namespace int8
} // namespace caffe2
#endif // CAFFE2_INT8_SIGMOID_OP_H_
