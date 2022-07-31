/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <altivec.h>

#include <qnnpack/x8zip.h>

// This function implements the operation ChannelShuffle with #groups=4
void pytorch_qnnp_x8zip_x4__vsx(size_t n, const void* input, void* output) {
  const uint8_t* x = input;
  const uint8_t* y = x + n;
  const uint8_t* z = y + n;
  const uint8_t* w = z + n;
  uint8_t* o = output;

  if (n >= 16) {
    do {
      const vector unsigned char vx = vec_xl(0, x);
      x += 16;
      const vector unsigned char vy = vec_xl(0, y);
      y += 16;
      const vector unsigned char vz = vec_xl(0, z);
      z += 16;
      const vector unsigned char vw = vec_xl(0, w);
      w += 16;
      const vector unsigned char vxy_hi = vec_mergeh(vx, vy);
      const vector unsigned char vxy_lo = vec_mergel(vx, vy);
      const vector unsigned char vzw_hi = vec_mergeh(vz, vw);
      const vector unsigned char vzw_lo = vec_mergel(vz, vw);
      const vector unsigned char vxyzw0 = (vector unsigned char)vec_mergeh(
          (vector unsigned short)vxy_hi, (vector unsigned short)vzw_hi);
      const vector unsigned char vxyzw1 = (vector unsigned char)vec_mergel(
          (vector unsigned short)vxy_hi, (vector unsigned short)vzw_hi);
      const vector unsigned char vxyzw2 = (vector unsigned char)vec_mergeh(
          (vector unsigned short)vxy_lo, (vector unsigned short)vzw_lo);
      const vector unsigned char vxyzw3 = (vector unsigned char)vec_mergel(
          (vector unsigned short)vxy_lo, (vector unsigned short)vzw_lo);
      vec_xst(vxyzw0, 0, o);
      vec_xst(vxyzw1, 16, o);
      vec_xst(vxyzw2, 32, o);
      vec_xst(vxyzw3, 48, o);
      o = (void*)((uintptr_t)o + 64);
      n -= 16;
    } while (n >= 16);
    if (n != 0) {
      const size_t address_increment = n - 16;
      const vector unsigned char vx = vec_xl(address_increment, x);
      const vector unsigned char vy = vec_xl(address_increment, y);
      const vector unsigned char vz = vec_xl(address_increment, z);
      const vector unsigned char vw = vec_xl(address_increment, w);
      const vector unsigned char vxy_hi = vec_mergeh(vx, vy);
      const vector unsigned char vxy_lo = vec_mergel(vx, vy);
      const vector unsigned char vzw_hi = vec_mergeh(vz, vw);
      const vector unsigned char vzw_lo = vec_mergel(vz, vw);
      const vector unsigned char vxyzw0 = (vector unsigned char)vec_mergeh(
          (vector unsigned short)vxy_hi, (vector unsigned short)vzw_hi);
      const vector unsigned char vxyzw1 = (vector unsigned char)vec_mergel(
          (vector unsigned short)vxy_hi, (vector unsigned short)vzw_hi);
      const vector unsigned char vxyzw2 = (vector unsigned char)vec_mergeh(
          (vector unsigned short)vxy_lo, (vector unsigned short)vzw_lo);
      const vector unsigned char vxyzw3 = (vector unsigned char)vec_mergel(
          (vector unsigned short)vxy_lo, (vector unsigned short)vzw_lo);
      o = (void*)((uintptr_t)o + address_increment * 4);
      vec_xst(vxyzw0, 0, o);
      vec_xst(vxyzw1, 16, o);
      vec_xst(vxyzw2, 32, o);
      vec_xst(vxyzw3, 48, o);
    }
  } else {
    do {
      const uint8_t vx = *x++;
      const uint8_t vy = *y++;
      const uint8_t vz = *z++;
      const uint8_t vw = *w++;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o[3] = vw;
      o += 4;
    } while (--n != 0);
  }
}
