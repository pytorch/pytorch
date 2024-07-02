/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <altivec.h>

#include <qnnpack/x8zip.h>

// This function implements the operation ChannelShuffle with #groups>4
void pytorch_qnnp_x8zip_xm__vsx(
    size_t n,
    size_t m,
    const void* input,
    void* output) {
  const uint8_t* w = input;
  const size_t input_increment = n * 3;
  const size_t output_increment = 4 - m * n;
  const uint8_t* last_input = w + n * (m - 1);
  void* last_output = (void*)((uintptr_t)output + (m - 4));

  const vector unsigned char vzero = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  if (n >= 8) {
    for (size_t i = 0; i < m; i += 4) {
      size_t k = n;
      w = (const uint8_t*)((uintptr_t)w + input_increment);
      if (w >= last_input) {
        w = last_input;
      }
      const uint8_t* z = (const uint8_t*)((uintptr_t)w - n);
      const uint8_t* y = (const uint8_t*)((uintptr_t)z - n);
      const uint8_t* x = (const uint8_t*)((uintptr_t)y - n);
      while (k >= 16) {
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

        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 0);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 1);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 2);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 3);
        output = (void*)((uintptr_t)output + m);

        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw1, 0);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw1, 1);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw1, 2);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw1, 3);
        output = (void*)((uintptr_t)output + m);

        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw2, 0);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw2, 1);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw2, 2);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw2, 3);
        output = (void*)((uintptr_t)output + m);

        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw3, 0);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw3, 1);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw3, 2);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw3, 3);
        output = (void*)((uintptr_t)output + m);

        k -= 16;
      }
      if (k >= 8) {
        const vector unsigned char vx = (vector unsigned char)vec_insert(
            ((uint64_t*)x)[0], (vector unsigned long long)vzero, 0);
        x += 8;
        const vector unsigned char vy = (vector unsigned char)vec_insert(
            ((uint64_t*)y)[0], (vector unsigned long long)vzero, 0);
        y += 8;
        const vector unsigned char vz = (vector unsigned char)vec_insert(
            ((uint64_t*)z)[0], (vector unsigned long long)vzero, 0);
        z += 8;
        const vector unsigned char vw = (vector unsigned char)vec_insert(
            ((uint64_t*)w)[0], (vector unsigned long long)vzero, 0);
        w += 8;

        const vector unsigned char vxy = vec_mergeh(vx, vy);
        const vector unsigned char vzw = vec_mergeh(vz, vw);
        const vector unsigned char vxyzw0 = (vector unsigned char)vec_mergeh(
            (vector unsigned short)vxy, (vector unsigned short)vzw);
        const vector unsigned char vxyzw1 = (vector unsigned char)vec_mergel(
            (vector unsigned short)vxy, (vector unsigned short)vzw);

        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 0);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 1);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 2);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 3);
        output = (void*)((uintptr_t)output + m);

        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw1, 0);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw1, 1);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw1, 2);
        output = (void*)((uintptr_t)output + m);
        *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw1, 3);
        output = (void*)((uintptr_t)output + m);

        k -= 8;
      }
      if (k != 0) {
        const size_t address_decrement = 8 - k;
        x -= address_decrement;
        y -= address_decrement;
        z -= address_decrement;
        w -= address_decrement;
        const vector unsigned char vshift = {
            8 * address_decrement, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        vector unsigned char vx = (vector unsigned char)vec_insert(
            ((uint64_t*)x)[0], (vector unsigned long long)vzero, 0);
        vector unsigned char vy = (vector unsigned char)vec_insert(
            ((uint64_t*)y)[0], (vector unsigned long long)vzero, 0);
        vector unsigned char vz = (vector unsigned char)vec_insert(
            ((uint64_t*)z)[0], (vector unsigned long long)vzero, 0);
        vector unsigned char vw = (vector unsigned char)vec_insert(
            ((uint64_t*)w)[0], (vector unsigned long long)vzero, 0);
        w += 8;

        vx = vec_sro(vx, vshift);
        vy = vec_sro(vy, vshift);
        vz = vec_sro(vz, vshift);
        vw = vec_sro(vw, vshift);

        const vector unsigned char vxy = vec_mergeh(vx, vy);
        const vector unsigned char vzw = vec_mergeh(vz, vw);
        vector unsigned char vxyzw0 = (vector unsigned char)vec_mergeh(
            (vector unsigned short)vxy, (vector unsigned short)vzw);
        vector unsigned char vxyzw1 = (vector unsigned char)vec_mergel(
            (vector unsigned short)vxy, (vector unsigned short)vzw);

        if (k & 4) {
          *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 0);
          output = (void*)((uintptr_t)output + m);
          *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 1);
          output = (void*)((uintptr_t)output + m);
          *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 2);
          output = (void*)((uintptr_t)output + m);
          *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 3);
          output = (void*)((uintptr_t)output + m);
          vxyzw0 = vxyzw1;
        }
        if (k & 2) {
          *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 0);
          output = (void*)((uintptr_t)output + m);
          *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 1);
          output = (void*)((uintptr_t)output + m);
          vxyzw0 = (vector unsigned char)vec_mergel(
              (vector unsigned long long)vxyzw0,
              (vector unsigned long long)vxyzw0);
        }
        if (k & 1) {
          *((uint32_t*)output) = vec_extract((vector unsigned int)vxyzw0, 0);
          output = (void*)((uintptr_t)output + m);
        }
      }
      output = (void*)((uintptr_t)output + output_increment);
      if (output > last_output) {
        output = last_output;
      }
    }
  } else {
    const uint8_t* i = input;
    uint8_t* o = output;
    size_t k = n;
    do {
      size_t l = m;
      const uint8_t* ii = i++;
      do {
        *o++ = *ii;
        ii += n;
      } while (--l != 0);
    } while (--k != 0);
  }
}
