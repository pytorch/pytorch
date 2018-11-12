#ifndef CAFFE2_OPERATORS_INT8_MAX_POOL_OP_H_
#define CAFFE2_OPERATORS_INT8_MAX_POOL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/conv_pool_op_base.h"
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

void Int8MaxPool(
    const uint8_t* input_data,
    at::IntList input_dims,
    int stride_width,
    int stride_height,
    int pad_width,
    int pad_height,
    int filter_width,
    int filter_height,
    uint8_t* output_data,
    at::IntList output_dims,
    uint8_t output_activation_min,
    uint8_t output_activation_max) {
  const int batches = input_dims[0];
  const int depth = input_dims[3];
  const int input_height = input_dims[1];
  const int input_width = input_dims[2];
  const int output_height = output_dims[1];
  const int output_width = output_dims[2];
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        const int in_y_origin = (out_y * stride_height) - pad_height;
        const int filter_x_start = std::max(0, -in_x_origin);
        const int filter_x_end =
            std::min(filter_width, input_width - in_x_origin);
        const int filter_y_start = std::max(0, -in_y_origin);
        const int filter_y_end =
            std::min(filter_height, input_height - in_y_origin);
        // 2048 required by Inception v3
        static constexpr int kAccBufferMaxSize = 2048;
        CHECK_LE(depth, kAccBufferMaxSize);
        uint8_t acc[kAccBufferMaxSize];
        memset(acc, 0, depth * sizeof(acc[0]));

        const uint8_t* input_ptr =
            &input_data
                [in_x_origin * depth + in_y_origin * input_width * depth +
                 batch * input_height * input_width * depth];

        for (int fy = filter_y_start; fy < filter_y_end; fy++) {
          const uint8_t* input_row_ptr =
              &input_ptr[fy * input_width * depth + filter_x_start * depth];

          for (int fx = filter_x_start; fx < filter_x_end; fx++) {
            int channel = 0;
#ifdef INT8_NEON_SIMD
            for (; channel <= depth - 16; channel += 16) {
              uint8x16_t acc_reg = vld1q_u8(acc + channel);
              uint8x16_t input_reg = vld1q_u8(input_row_ptr);
              input_row_ptr += 16;
              acc_reg = vmaxq_u8(acc_reg, input_reg);
              vst1q_u8(acc + channel, acc_reg);
            }

            for (; channel <= depth - 8; channel += 8) {
              uint8x8_t acc_reg = vld1_u8(acc + channel);
              uint8x8_t input_reg = vld1_u8(input_row_ptr);
              input_row_ptr += 8;
              acc_reg = vmax_u8(acc_reg, input_reg);
              vst1_u8(acc + channel, acc_reg);
            }
#endif
            for (; channel < depth; ++channel) {
              acc[channel] = std::max(acc[channel], *input_row_ptr++);
            }
          }
        }
        uint8_t* output_ptr =
            &output_data
                [out_x * depth + out_y * output_width * depth +
                 batch * output_height * output_width * depth];
        int channel = 0;
#ifdef INT8_NEON_SIMD
        for (; channel <= depth - 16; channel += 16) {
          uint8x16_t a = vld1q_u8(acc + channel);
          a = vminq_u8(a, vdupq_n_u8(output_activation_max));
          a = vmaxq_u8(a, vdupq_n_u8(output_activation_min));
          vst1q_u8(output_ptr + channel, a);
        }
        for (; channel <= depth - 8; channel += 8) {
          uint8x8_t a = vld1_u8(acc + channel);
          a = vmin_u8(a, vdup_n_u8(output_activation_max));
          a = vmax_u8(a, vdup_n_u8(output_activation_min));
          vst1_u8(output_ptr + channel, a);
        }
#endif
        for (; channel < depth; ++channel) {
          uint8_t a = acc[channel];
          a = std::max<uint8_t>(a, output_activation_min);
          a = std::min<uint8_t>(a, output_activation_max);
          output_ptr[channel] = static_cast<uint8_t>(a);
        }
      }
    }
  }
}

} // namespace

template <Activation Ac>
class Int8MaxPoolOp final : public ConvPoolOpBase<CPUContext> {
 public:
  Int8MaxPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        this->order_ == StorageOrder::NHWC, "Int8 only supports NCHW order.");
  }

  bool RunOnDeviceWithOrderNHWC() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    Y->scale = X.scale;
    Y->zero_point = X.zero_point;
    const int32_t Y_offset =
        this->template GetSingleArgument<int>("Y_zero_point", 0);
    const float Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    CHECK_EQ(Y_offset, X.zero_point);
    CHECK_EQ(Y_scale, X.scale);

    CHECK_EQ(X.t.dim(), 4);
    const int height = X.t.dim32(1);
    const int width = X.t.dim32(2);
    const int channels = X.t.dim32(3);
    ConvPoolOpBase<CPUContext>::SetOutputSize(X.t, &(Y->t), channels);

    Int8MaxPool(
        X.t.template data<uint8_t>(),
        X.t.sizes(),
        stride_w(),
        stride_h(),
        pad_l(),
        pad_t(),
        kernel_w(),
        kernel_h(),
        Y->t.template mutable_data<uint8_t>(),
        Y->t.sizes(),
        activationLimits(Y->scale, Y->zero_point, Ac).first,
        activationLimits(Y->scale, Y->zero_point, Ac).second);
    return true;
  }
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_MAX_POOL_OP_H_
