
#pragma once

#include "arm_neon_support.h"

void interleaveSlice(void* output,
                     const float* input,
                     size_t width,
                     size_t height,
                     size_t row_stride,
                     uint16_t input_channels);
void deInterleaveSlice(float* output,
                       const void* input,
                       size_t width,
                       size_t height,
                       size_t input_stride,
                       uint32_t output_channels);
