#ifndef CAFFE2_OPERATORS_INT8_SOFTMAX_OP_H_
#define CAFFE2_OPERATORS_INT8_SOFTMAX_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
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

void PreprocessSoftmaxScaling(
    double beta,
    double input_scale,
    int input_integer_bits,
    int32_t* quantized_multiplier,
    int* left_shift) {
  // If the overall multiplier (input and beta) is large, then exp() of an
  // input difference of 1 scaled by this will be large.  In other words, we
  // can cap the multiplier and know that, when it is used, the output will be
  // (round to) zero wherever the input is not at the maximum value.

  // If the overall scale is less than one, and input_integer_bits=0, then the
  // result is double equivalent of Q0.31 (actually with more precision). Thus
  // this generates a Q(input_integer_bits).(31-input_integer_bits)
  // representation.
  const double input_beta_real_multiplier = std::min(
      beta * input_scale * (1 << (31 - input_integer_bits)), (1ll << 31) - 1.0);

  QuantizeMultiplierGreaterThanOne(
      input_beta_real_multiplier, quantized_multiplier, left_shift);
}

void Int8Softmax(
    const uint8_t* input_data,
    const size_t N,
    const size_t D,
    int32_t input_beta_multiplier,
    int32_t input_beta_left_shift,
    int diff_min,
    uint8_t* output_data) {
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static const int kScaledDiffIntegerBits = 5;
  static const int kAccumulationIntegerBits = 12;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32_t, kScaledDiffIntegerBits>;
  using FixedPointAccum =
      gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;

  for (int n = 0; n < N; ++n) {
    uint8_t max_in_row = 0;
    for (int c = 0; c < D; ++c) {
      max_in_row = std::max(max_in_row, input_data[n * D + c]);
    }

    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    for (int c = 0; c < D; ++c) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[n * D + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        sum_of_exps = sum_of_exps +
            gemmlowp::Rescale<kAccumulationIntegerBits>(
                          exp_on_negative_values(scaled_diff_f8));
      }
    }

    int32_t fixed_sum_of_exps = sum_of_exps.raw();
    // TODO(starka): Use a NEON intrinsic like vclzq_u32 instead.
    int headroom_plus_one =
        __builtin_clz(static_cast<uint32_t>(fixed_sum_of_exps));
    // This is the number of bits to the left of the binary point above 1.0.
    // Consider fixed_sum_of_exps=1.25.  In that case shifted_scale=0.8 and
    // no later adjustment will be needed.
    int num_bits_over_unit = kAccumulationIntegerBits - headroom_plus_one;
    int32_t shifted_sum_minus_one = static_cast<int32_t>(
        (static_cast<uint32_t>(fixed_sum_of_exps) << headroom_plus_one) -
        (static_cast<uint32_t>(1) << 31));

    FixedPoint0 shifted_scale;
    // gemmlowp::one_over_one_plus_x_for_x_in_0_1 is defined on (0,
    // 1), not [0, 1), so need to handle the case where
    // shifted_sum_minus_one is exactly 0.
    if (shifted_sum_minus_one == 0) {
      shifted_scale = FixedPoint0::One();
    } else {
      shifted_scale = gemmlowp::one_over_one_plus_x_for_x_in_0_1(
          FixedPoint0::FromRaw(shifted_sum_minus_one));
    }

    for (int c = 0; c < D; ++c) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[n * D + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);

        FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
        int32_t unsat_output = gemmlowp::RoundingDivideByPOT(
            (shifted_scale * exp_in_0).raw(), num_bits_over_unit + 31 - 8);

        output_data[n * D + c] = std::max(std::min(unsat_output, 255), 0);

      } else {
        output_data[n * D + c] = 0;
      }
    }
  }
}

} // namespace

class Int8SoftmaxOp final : public Operator<CPUContext> {
 public:
  Int8SoftmaxOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    const int32_t Y_offset =
        this->template GetSingleArgument<int>("Y_zero_point", 0);
    const float Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    CHECK_EQ(Y_offset, 0);
    CHECK_EQ(Y_scale, 1. / 256);

    static const int kScaledDiffIntegerBits = 5;
    Y->scale = Y_scale;
    Y->zero_point = Y_offset;
    Y->t.ResizeLike(X.t);
    int32_t input_multiplier;
    int input_left_shift;
    PreprocessSoftmaxScaling(
        1.0 /*params->beta*/,
        X.scale,
        kScaledDiffIntegerBits,
        &input_multiplier,
        &input_left_shift);
    const int diff_min =
        -1.0 * CalculateInputRadius(kScaledDiffIntegerBits, input_left_shift);
    Int8Softmax(
        X.t.data<uint8_t>(),
        X.t.size(0),
        X.t.numel() / X.t.size(0),
        input_multiplier,
        input_left_shift,
        diff_min,
        Y->t.mutable_data<uint8_t>());
    return true;
  }
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_SOFTMAX_OP_H_
