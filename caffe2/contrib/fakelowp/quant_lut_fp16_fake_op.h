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

    constexpr int tanhLUTMinOffset = 7000;
    constexpr int tanhLUTMaxOffset = 18000;
    constexpr int lutSize = tanhLUTMaxOffset - tanhLUTMinOffset;

    std::array<uint8_t, lutSize> tanhLUT;

    Y_scale = 1.0f / Y_scale;

/*
        for (int i = 0; i < size; i++)
        {
            short startOfLut = i + TANH_FP16_U8_LUT_MIN_OFFSET;
            half input = (*((half*)&startOfLut));
            float x = input;
            float tanh_x = tanh(x);
            unsigned int tanh_quant = (unsigned int)round(float(tanh_x * params.outputscale - params.outputoffset));
            if (tanh_quant > 255) // clamp to max
            {
                tanh_quant = 255;
            }
            Lut[i] = (uint8_t)tanh_quant;
        }
*/



    // create table once
    for (int i = 0; i < lutSize; i++)
    {
        short input = i + tanhLUTMinOffset;
        float x = _cvtsh_ss(input);
        float tanh_x = tanh(x);
        float pre = tanh_x;
        tanh_x = tanh_x * Y_scale + Y_offset;

        if (tanh_x > 255) {
            tanh_x = 255;
        }
        tanh_x = round(tanh_x);
        LOG_FIRST_N(INFO,50) << i << " " << input << " " << x << " " \
                             << pre << " " << tanh_x << " " << Y_scale << " " << Y_offset;

        if (tanh_x < 0) {
            tanh_x = 0;
            LOG_FIRST_N(WARNING, 10) << x << " gets a  negative tanh";
        }

        uint32_t tanh_quant = (uint32_t)round(tanh_x);

        tanhLUT[i] = (uint8_t)tanh_quant;
    }

    const float* X_data = X.template data<float>();
    for (int i = 0; i < X.numel(); i++) {
        float val = X_data[i];
        short shortAbsInput = _cvtss_sh(abs(val), 0);
        short clampShortAbsInput = std::clamp(shortAbsInput, (short)tanhLUTMinOffset, (short)(tanhLUTMaxOffset - 1));
        short inputInLutRange = clampShortAbsInput - tanhLUTMinOffset;
        short temp =  tanhLUT[inputInLutRange];

        LOG_FIRST_N(INFO,100) << val << " " << shortAbsInput << " " << clampShortAbsInput << " " << inputInLutRange << " " << temp;


        if (val < 0.0) // Handle negative input
        {
            temp = temp - Y_offset;
            temp = temp * (-1);
            temp = temp + Y_offset;
        }
        uint8_t output = (uint8_t)temp;
        if (temp < 0)
        {
            output = 0;
        }
         Y->t.mutable_data<uint8_t>()[i] = output;
    }

    return true;
  }
};

}
}

/*

 for (int i = 0; i < inputTensor.GetNumElements(); i++) {
            half val = inputData[i];
            half absInput = ABS_HALF(val);
            short shortAbsInput = (*((short*)&absInput));
            short clampShortAbsInput = clamp(shortAbsInput, (short)TANH_FP16_U8_LUT_MIN_OFFSET, (short)(TANH_FP16_U8_LUT_MAX_OFFSET-1));
            short inputInLutRange = clampShortAbsInput - TANH_FP16_U8_LUT_MIN_OFFSET;
            short temp =  lut[inputInLutRange];
            if (val < 0.0) // Handle negative input
            {
                temp = temp + offset;
                temp = temp * (-1);
                temp = temp - offset;
            }
            uint8_t output = (uint8_t)temp;
            if (temp < 0)
            {
                output = 0;
            }
            outputData[i] = output;
        }




#define TANH_FP16_U8_LUT_MIN_OFFSET 7000
#define TANH_FP16_U8_LUT_MAX_OFFSET 18000
#define TANH_FP16_U8_LUT_SIZE (TANH_FP16_U8_LUT_MAX_OFFSET - TANH_FP16_U8_LUT_MIN_OFFSET)

 void  CreateTanhFp16toU8Lut(int size, uint8_t * Lut, const LutParams& params) {

        GT_THROW_ERROR_IF(size != TANH_FP16_U8_LUT_SIZE, NNPI_INTERNAL_ERROR, "wrong LUT size");
        for (int i = 0; i < size; i++)
        {
            short startOfLut = i + TANH_FP16_U8_LUT_MIN_OFFSET;
            half input = (*((half*)&startOfLut));
            float x = input;
            float tanh_x = tanh(x);
            unsigned int tanh_quant = (unsigned int)round(float(tanh_x * params.outputscale - params.outputoffset));
            if (tanh_quant > 255) // clamp to max
            {
                tanh_quant = 255;
            }
            Lut[i] = (uint8_t)tanh_quant;
        }
    }
*/
