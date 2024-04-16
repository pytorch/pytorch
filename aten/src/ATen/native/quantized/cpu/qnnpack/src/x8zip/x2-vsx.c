/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <altivec.h>

#include <qnnpack/x8zip.h>

// This function implements the operation ChannelShuffle with #groups=2
void pytorch_qnnp_x8zip_x2__vsx(size_t n, const void* input, void* output) {
  const uint8_t* x = input;
  const uint8_t* y = x + n;
  uint8_t* o = output;
  if (n >= 16) {
    do {
      const vector unsigned char vx = vec_xl(0, x);
      x += 16;
      const vector unsigned char vy = vec_xl(0, y);
      y += 16;
      const vector unsigned char vxy_hi = vec_mergeh(vx, vy);
      const vector unsigned char vxy_lo = vec_mergel(vx, vy);
      vec_xst(vxy_hi, 0, o);
      vec_xst(vxy_lo, 16, o);
      o = (void*)((uintptr_t)o + 32);
      n -= 16;
    } while (n >= 16);
    if (n != 0) {
      const size_t address_increment = n - 16;
      const vector unsigned char vx = vec_xl(address_increment, x);
      const vector unsigned char vy = vec_xl(address_increment, y);
      const vector unsigned char vxy_hi = vec_mergeh(vx, vy);
      const vector unsigned char vxy_lo = vec_mergel(vx, vy);
      o = (void*)((uintptr_t)o + address_increment * 2);
      vec_xst(vxy_hi, 0, o);
      vec_xst(vxy_lo, 16, o);
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
