/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <altivec.h>

#include <qnnpack/u8rmax.h>

// This function returns the maximum element in x
uint8_t pytorch_u8rmax_ukernel__vsx(size_t n, const uint8_t* x) {
  assert(n != 0);

  if
    PYTORCH_QNNP_LIKELY(n >= 16) {
      vector unsigned char vmax = {
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      do {
        const vector unsigned char vx = vec_xl(0, x);
        x += 16;
        vmax = vec_max(vmax, vx);
        n -= 16;
      } while (n >= 16);
      if (n != 0) {
        const size_t x_increment = n - 16;
        x = (const uint8_t*)((uintptr_t)x + x_increment);
        const vector unsigned char vx = vec_xl(0, x);
        vmax = vec_max(vmax, vx);
      }
      const vector unsigned char vshift_4bytes = {
          32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      const vector unsigned char vshift_2bytes = {
          16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      const vector unsigned char vshift_1byte = {
          8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      vmax = vec_max(
          vmax,
          (vector unsigned char)vec_mergel(
              (vector unsigned long long)vmax,
              (vector unsigned long long)vmax));
      vmax = vec_max(vmax, vec_sro(vmax, vshift_4bytes));
      vmax = vec_max(vmax, vec_sro(vmax, vshift_2bytes));
      vmax = vec_max(vmax, vec_sro(vmax, vshift_1byte));
      return vmax[0];
    }
  else {
    // Scalar execution
    uint8_t vmax = 0;
    do {
      const uint8_t vx = *x++;
      vmax = vx > vmax ? vx : vmax;
    } while (--n != 0);
    return vmax;
  }
}
