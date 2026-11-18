/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>

#include <qnnpack/x8zip.h>

void pytorch_qnnp_x8zip_x2__neon(size_t n, const void* input, void* output) {
  const uint8_t* x = input;
  const uint8_t* y = x + n;
  uint8_t* o = output;

  if (n >= 8) {
    do {
      uint8x8x2_t vxy;
      vxy.val[0] = vld1_u8(x);
      x += 8;
      vxy.val[1] = vld1_u8(y);
      y += 8;
      vst2_u8(o, vxy);
      o += 16;
      ;
      n -= 8;
    } while (n >= 8);
    if (n != 0) {
      const size_t address_increment = n - 8;
      uint8x8x2_t vxy;
      vxy.val[0] = vld1_u8((const uint8_t*)((uintptr_t)x + address_increment));
      vxy.val[1] = vld1_u8((const uint8_t*)((uintptr_t)y + address_increment));
      vst2_u8((uint8_t*)((uintptr_t)o + address_increment * 2), vxy);
    }
  } else {
    do {
      const uint8_t vx = *x++;
      const uint8_t vy = *y++;
      o[0] = vx;
      o[1] = vy;
      o += 2;
    } while (--n != 0);
  }
}
