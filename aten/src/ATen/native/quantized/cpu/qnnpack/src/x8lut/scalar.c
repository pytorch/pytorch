/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <qnnpack/x8lut.h>

void pytorch_x8lut_ukernel__scalar(
    size_t n,
    const uint8_t* x,
    const uint8_t t[RESTRICT_STATIC 256],
    uint8_t* y) {
  assert(n != 0);

  while (n >= 4) {
    const size_t vx0 = x[0];
    const size_t vx1 = x[1];
    const size_t vx2 = x[2];
    const size_t vx3 = x[3];
    x += 4;

    const uint8_t vt0 = t[vx0];
    const uint8_t vt1 = t[vx1];
    const uint8_t vt2 = t[vx2];
    const uint8_t vt3 = t[vx3];

    y[0] = vt0;
    y[1] = vt1;
    y[2] = vt2;
    y[3] = vt3;
    y += 4;

    n -= 4;
  }
  while (n != 0) {
    const size_t vx = *x++;
    const uint8_t vt = t[vx];
    *y++ = vt;

    n--;
  };
}
