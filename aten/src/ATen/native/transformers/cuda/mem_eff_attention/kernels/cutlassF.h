/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#pragma once
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
// ======== bf16 / sm80 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x64_rf_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x128_rf_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_bf16_sm80(T cb, int cc) {
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>(), fmha_cutlassF_bf16_aligned_64x64_rf_sm80);
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>(), fmha_cutlassF_bf16_aligned_64x128_rf_sm80);
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>(), fmha_cutlassF_bf16_aligned_32x128_gmem_sm80);
}

// ======== f16 / sm50 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_64x64_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_rf_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_gmem_sm50(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f16_sm50(T cb, int cc) {
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_32x128_rf_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 64, 64, 64, true, true>(), fmha_cutlassF_f16_notaligned_64x64_rf_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 128, true, true>(), fmha_cutlassF_f16_notaligned_32x128_rf_sm50);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_notaligned_32x128_gmem_sm50);
}

// ======== f16 / sm70 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_64x64_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_rf_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_gmem_sm70(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f16_sm70(T cb, int cc) {
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_32x128_rf_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 64, 64, 64, true, true>(), fmha_cutlassF_f16_notaligned_64x64_rf_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 128, true, true>(), fmha_cutlassF_f16_notaligned_32x128_rf_sm70);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_notaligned_32x128_gmem_sm70);
}

// ======== f16 / sm75 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_64x64_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_rf_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_notaligned_32x128_gmem_sm75(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f16_sm75(T cb, int cc) {
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm75);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_32x128_rf_sm75);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm75);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 64, 64, 64, true, true>(), fmha_cutlassF_f16_notaligned_64x64_rf_sm75);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 128, true, true>(), fmha_cutlassF_f16_notaligned_32x128_rf_sm75);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_notaligned_32x128_gmem_sm75);
}

// ======== f16 / sm80 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x128_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f16_sm80(T cb, int cc) {
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm80);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>(), fmha_cutlassF_f16_aligned_64x128_rf_sm80);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm80);
}

// ======== f32 / sm50 ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_rf_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_gmem_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_64x64_rf_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, false, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_rf_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_gmem_sm50(typename AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f32_sm50(T cb, int cc) {
    cb(AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm50);
    cb(AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_32x128_rf_sm50);
    cb(AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm50);
    cb(AttentionKernel<float, cutlass::arch::Sm50, false, 64, 64, 64, true, true>(), fmha_cutlassF_f32_notaligned_64x64_rf_sm50);
    cb(AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 128, true, true>(), fmha_cutlassF_f32_notaligned_32x128_rf_sm50);
    cb(AttentionKernel<float, cutlass::arch::Sm50, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_notaligned_32x128_gmem_sm50);
}

// ======== f32 / sm70 ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_rf_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_gmem_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_64x64_rf_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, false, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_rf_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_gmem_sm70(typename AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f32_sm70(T cb, int cc) {
    cb(AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_32x128_rf_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, false, 64, 64, 64, true, true>(), fmha_cutlassF_f32_notaligned_64x64_rf_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 128, true, true>(), fmha_cutlassF_f32_notaligned_32x128_rf_sm70);
    cb(AttentionKernel<float, cutlass::arch::Sm70, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_notaligned_32x128_gmem_sm70);
}

// ======== f32 / sm75 ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_rf_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_gmem_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_64x64_rf_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, false, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_rf_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_notaligned_32x128_gmem_sm75(typename AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f32_sm75(T cb, int cc) {
    cb(AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_32x128_rf_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, false, 64, 64, 64, true, true>(), fmha_cutlassF_f32_notaligned_64x64_rf_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 128, true, true>(), fmha_cutlassF_f32_notaligned_32x128_rf_sm75);
    cb(AttentionKernel<float, cutlass::arch::Sm75, false, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_notaligned_32x128_gmem_sm75);
}

// ======== f32 / sm80 ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x128_rf_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_gmem_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f32_sm80(T cb, int cc) {
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm80);
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>(), fmha_cutlassF_f32_aligned_64x128_rf_sm80);
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm80);
}


template <typename DT, typename T>
void dispatch_cutlassF(T cb, int cc = 0) {

    if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassF_bf16_sm80(cb, cc);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 50 <= cc && cc < 70) {
        dispatch_cutlassF_f16_sm50(cb, cc);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 70 <= cc && cc < 75) {
        dispatch_cutlassF_f16_sm70(cb, cc);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 75 <= cc && cc < 80) {
        dispatch_cutlassF_f16_sm75(cb, cc);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassF_f16_sm80(cb, cc);
    }
    if (std::is_same<DT, float>::value && 50 <= cc && cc < 70) {
        dispatch_cutlassF_f32_sm50(cb, cc);
    }
    if (std::is_same<DT, float>::value && 70 <= cc && cc < 75) {
        dispatch_cutlassF_f32_sm70(cb, cc);
    }
    if (std::is_same<DT, float>::value && 75 <= cc && cc < 80) {
        dispatch_cutlassF_f32_sm75(cb, cc);
    }
    if (std::is_same<DT, float>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassF_f32_sm80(cb, cc);
    }
}
