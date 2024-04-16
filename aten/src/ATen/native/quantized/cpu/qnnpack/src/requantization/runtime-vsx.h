/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <altivec.h>

PYTORCH_QNNP_INLINE vector short
sub_zero_point(const vector short va, const vector short vzp) {
#if PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
  // Run-time quantization
  return vec_sub(va, vzp);
#else
  // Design-time quantization (no-op)
  return va;
#endif
}
