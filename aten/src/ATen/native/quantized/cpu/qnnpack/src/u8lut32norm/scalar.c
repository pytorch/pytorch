/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <fxdiv.h>

#include <qnnpack/u8lut32norm.h>

static inline uint32_t compute_sum(
    size_t n,
    const uint8_t* x,
    const uint32_t* t) {
  assert(n != 0);

  uint32_t vsum = 0;
  do {
    const size_t vx = *x++;
    vsum += t[vx];
  } while (--n != 0);
  return vsum;
}

void pytorch_u8lut32norm_ukernel__scalar(
    size_t n,
    const uint8_t* x,
    const uint32_t* t,
    uint8_t* y) {
  assert(n != 0);

  const uint32_t vsum = compute_sum(n, x, t);
  assert(vsum != 0);

  struct fxdiv_divisor_uint32_t vsum_divisor = fxdiv_init_uint32_t(vsum);
  const uint32_t vrounding = (vsum >> 1);
  do {
    const size_t vx = *x++;
    const uint32_t vt = t[vx];
    const uint32_t vq =
        fxdiv_quotient_uint32_t((vt << 8) + vrounding, vsum_divisor);
    const uint8_t vy = vq > 255 ? UINT8_C(255) : (uint8_t)vq;
    *y++ = vy;
  } while (--n != 0);
}
