/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <immintrin.h>

PYTORCH_QNNP_INLINE __m128i
sub_zero_point(const __m128i va, const __m128i vzp) {
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
  // Run-time quantization
  return _mm_sub_epi16(va, vzp);
#else
  // Design-time quantization (no-op)
  return va;
#endif
}
