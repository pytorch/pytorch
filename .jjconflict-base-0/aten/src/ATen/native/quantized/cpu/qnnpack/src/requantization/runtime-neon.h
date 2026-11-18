/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arm_neon.h>

PYTORCH_QNNP_INLINE uint16x8_t
sub_zero_point(const uint8x8_t va, const uint8x8_t vzp) {
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
  // Run-time quantization
  return vsubl_u8(va, vzp);
#else
  // Design-time quantization
  return vmovl_u8(va);
#endif
}
