/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <altivec.h>

#include <qnnpack/x8zip.h>

// This function implements the operation ChannelShuffle with #groups=3
void pytorch_qnnp_x8zip_x3__vsx(size_t n, const void* input, void* output) {

  const uint8_t* x = input;
  const uint8_t* y = x + n;
  const uint8_t* z = y + n;
  uint8_t* o = output;

  if (n >= 16) {
    // x0 y0 0 x1 y1 0 x2 y2 0 x3 y3 0 x4 y4 0 x5
    const vector unsigned char mask_p1_1 = {
        0, 16, 0, 1, 17, 0, 2, 18, 0, 3, 19, 0, 4, 20, 0, 5};
    // x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 x5
    const vector unsigned char mask_p1_2 = {
        0, 1, 16, 3, 4, 17, 6, 7, 18, 9, 10, 19, 12, 13, 20, 15};
    // y5 0 x6 y6 0 x7 y7 0 x8 y8 0 x9 y9 0 x10 y10
    const vector unsigned char mask_p2_1 = {
        21, 0, 6, 22, 0, 7, 23, 0, 8, 24, 0, 9, 25, 0, 10, 26};
    // y5 z5 x6 y6 z6 x7 y7 z7 x8 y8 z8 x9 y9 z9 x10 y10
    const vector unsigned char mask_p2_2 = {
        0, 21, 2, 3, 22, 5, 6, 23, 8, 9, 24, 11, 12, 25, 14, 15};
    // 0 x11 y11 0 x12 y12 0 x13 y13 0 x14 y14 0 x15 y15 0
    const vector unsigned char mask_p3_1 = {
        0, 11, 27, 0, 12, 28, 0, 13, 29, 0, 14, 30, 0, 15, 31, 0};
    // z10 x11 y11 z11 x12 y12 z12 x13 y13 z13 x14 y14 z14 x15 y15 z15
    const vector unsigned char mask_p3_2 = {
        26, 1, 2, 27, 4, 5, 28, 7, 8, 29, 10, 11, 30, 13, 14, 31};
    do {
      const vector unsigned char vx = vec_xl(0, x);
      x += 16;
      const vector unsigned char vy = vec_xl(0, y);
      y += 16;
      const vector unsigned char vz = vec_xl(0, z);
      z += 16;
      vector unsigned char t0 = vec_perm(vx, vy, mask_p1_1);
      vector unsigned char t1 = vec_perm(vx, vy, mask_p2_1);
      vector unsigned char t2 = vec_perm(vx, vy, mask_p3_1);
      vector unsigned char vxyz0 = vec_perm(t0, vz, mask_p1_2);
      vector unsigned char vxyz1 = vec_perm(t1, vz, mask_p2_2);
      vector unsigned char vxyz2 = vec_perm(t2, vz, mask_p3_2);
      vec_xst(vxyz0, 0, o);
      vec_xst(vxyz1, 16, o);
      vec_xst(vxyz2, 32, o);
      o += 48;
      n -= 16;
    } while (n >= 16);
    if (n != 0) {
      const size_t address_increment = n - 16;
      const vector unsigned char vx = vec_xl(address_increment, x);
      const vector unsigned char vy = vec_xl(address_increment, y);
      const vector unsigned char vz = vec_xl(address_increment, z);
      vector unsigned char t0 = vec_perm(vx, vy, mask_p1_1);
      vector unsigned char t1 = vec_perm(vx, vy, mask_p2_1);
      vector unsigned char t2 = vec_perm(vx, vy, mask_p3_1);
      vector unsigned char vxyz0 = vec_perm(t0, vz, mask_p1_2);
      vector unsigned char vxyz1 = vec_perm(t1, vz, mask_p2_2);
      vector unsigned char vxyz2 = vec_perm(t2, vz, mask_p3_2);
      o = (void*)((uintptr_t)o + address_increment * 3);
      vec_xst(vxyz0, 0, o);
      vec_xst(vxyz1, 16, o);
      vec_xst(vxyz2, 32, o);
    }
  } else {
    do {
      const uint8_t vx = *x++;
      const uint8_t vy = *y++;
      const uint8_t vz = *z++;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o += 3;
    } while (--n != 0);
  }
}
