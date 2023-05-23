/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x64_k96_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 900
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `fmha_cutlassB_bf16_aligned_128x64_k96_sm80` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
