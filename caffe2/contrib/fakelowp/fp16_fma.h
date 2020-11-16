#pragma once
#include <glog/logging.h>

namespace fake_fp16 {

// Compute FMA using fp16 accumulation
// Out = FMA (A, B, Out)
void fma_fp16(int N, const float* A, const float* B, float* Out);

void fma_fp16_slow(int N, const float* A, const float* B, float* Out);

float fma_fp16_slow(const float A, const float B, float Out);

float fmafp32_avx_emulation(float v1, float v2, float v3);

} // namespace fake_fp16
