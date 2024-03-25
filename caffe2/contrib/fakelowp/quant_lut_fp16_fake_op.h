#pragma once

#include <array>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"

#include <immintrin.h>
#include <emmintrin.h>


namespace caffe2 {

namespace {


class TanhInt8QuantizeNNPIOp final : public Operator<CPUContext> {
 public:
  using Operator<CPUContext>::Operator;

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Outputs()[0]->template GetMutable<int8::Int8TensorCPU>();
    Y->t.ResizeLike(X);

    int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
    auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);

    Y->scale = Y_scale;
    Y->zero_point = Y_offset;

    constexpr int tanhLUTMinOffset = 0;
    constexpr int tanhLUTMaxOffset = 18000;
    constexpr int lutSize = tanhLUTMaxOffset - tanhLUTMinOffset;

    std::array<uint8_t, lutSize> tanhLUT;

    Y_scale = 1.0f / Y_scale;

    // create table once
    for (const auto i : c10::irange(lutSize)) {
        short input = i + tanhLUTMinOffset;
        float x = _cvtsh_ss(input);
        float tanh_x = tanh(x);
        tanh_x = round(tanh_x * Y_scale + Y_offset);

        if (tanh_x < 0 || tanh_x > 255.0) {
            tanh_x = 255.0;
        }
        uint32_t tanh_quant = (uint32_t)(tanh_x);

        tanhLUT[i] = (uint8_t)tanh_quant;
    }

    const float* X_data = X.template data<float>();
    for (const auto i : c10::irange(X.numel())) {
        short val = _cvtss_sh(X_data[i], 0);
        unsigned short max16BitPositive = 0x7FFF;
        unsigned short input16Bit = (*(unsigned short*)& val);
        short shortAbsInput = input16Bit & max16BitPositive; // mask out negative bit
        short clampShortAbsInput = shortAbsInput;
        if (shortAbsInput < (short)tanhLUTMinOffset) {
            clampShortAbsInput = (short)tanhLUTMinOffset;
        }

        if (shortAbsInput > (short)(tanhLUTMaxOffset - 1)) {
            clampShortAbsInput = (short)(tanhLUTMaxOffset - 1);
        }
        short inputInLutRange = clampShortAbsInput - tanhLUTMinOffset;
        short temp =  tanhLUT[inputInLutRange];

        if (input16Bit > max16BitPositive) {  // negative value
            temp = temp - Y_offset;
            temp = temp * (-1);
            temp = temp + Y_offset;
        }
        uint8_t output = (uint8_t)temp;
        if (temp < 0) {
            output = 0;
        }

        Y->t.mutable_data<uint8_t>()[i] = output;
    }

    return true;
  }
};

}
}
