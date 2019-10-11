/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <limits.h>
#include <stdint.h>

/*
 * The code below is adapted from Google's gemmlowp library.
 * It is only used in QNNPACK unit tests and comparative benchmarks,
 * but not the library itself.
 */

// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

inline static int32_t gemmlowp_scalar_vqrdmulh_s32(int32_t a, int32_t b) {
  const bool overflow = a == b && a == INT32_MIN;
  const int64_t ab_64 = (int64_t)a * (int64_t)b;
  const int32_t nudge =
      (a ^ b) >= 0 ? INT32_C(0x40000000) : -INT32_C(0x3FFFFFFF);
  const int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) / INT64_C(0x80000000));
  return overflow ? INT32_MAX : ab_x2_high32;
}

inline static int32_t gemmlowp_scalar_rdivbypo2_s32(int32_t x, int exponent) {
  const int32_t mask = ((1 << exponent) - 1);
  const int32_t remainder = x & mask;
  const int32_t threshold = (mask >> 1) + (int32_t)(x < 0);
  return asr_s32(x, exponent) + (int32_t)(remainder > threshold);
}
