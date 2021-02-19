/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <qnnpack/assembly.h>

# Packed A format.
# 8kx4m blocks for alls blocks given 4 rows (4m) are placed in contiguous memory.
# Original A
# --------- K -----------          -- (K + 4 - 1) / 4 --
# |                     |          |                   |
# |                     |        (M + 8 - 1)/8         |
# |                     | Packed   |                   |
# M                     |  =>      |-------------------|
# |                     |        Thus Packed A has (K + 4 - 1)/4 * (M + 8 -1)/8 blocks
# |                     |
# |---------------------|
# 
# Each 8 x 4 blocks is transposed and stored.
# Each of the (K + 4 - 1)/4 blocks for a given group of 8 m blocks
# are stored adjacent in memory
# Thus, each block:
# |----8m-----|----8m-----|
# 4k          |           | ..... (K + 4 - 1)/4 blocks
# |-----------|-----------|
# This locality helps in loading 8kx8m blocks of activations
# Note when M is not multiple of 8, the rest can contain arbitrary
# data in packed A as we will not be writing those out.
# This wil be taken care by just copying the appropriate valid data

# void pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch32_neon(
#     size_t mr,
#     size_t K,
#     const uint8_t* a,
#     size_t a_stride,
#     uint8_t* packed_a,
BEGIN_FUNCTION pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon

    # x2 = a0 = a pointer
    # x4 = packed_a pointer

    CMP x0, 2
    # x5 = a1
    ADD x5, x2, x3
    CSEL x5, x2, x5, LO

    # x6 = a2
    ADD x6, x5, x3
    CSEL x6, x5, x6, LS

    CMP x0, 4
    # x7 = a3
    ADD x7, x6, x3
    CSEL x7, x6, x7, LO

    # x8 = a4
    ADD x8, x7, x3
    CSEL x8, x7, x8, LS

    CMP x0, 6
    # x9 = a5
    ADD x9, x8, x3
    CSEL x9, x8, x9, LO

    # x10 = a6
    ADD x10, x9, x3
    CSEL x10, x9, x10, LS

    CMP x0, 8
    # x11 = a7
    ADD x11, x10, x3
    CSEL x11, x10, x11, NE

    # num_k_blocks = (k + (4 - 1)) / 4
    ADD x1, x1, 3
    LSR x1, x1, 2

    SUBS x1, x1, 2
    B.LO 1f

    .p2align 5
k_loop:
    LD1 {v0.d}[0], [x2], 8
    LD1 {v0.d}[1], [x8], 8
    LD1 {v1.d}[0], [x5], 8
    LD1 {v1.d}[1], [x9], 8
    LD1 {v2.d}[0], [x6], 8
    LD1 {v2.d}[1], [x10], 8
    LD1 {v3.d}[0], [x7], 8
    LD1 {v3.d}[1], [x11], 8

    #  Now we have 8x8 block of values that we will tranpose
    #  A matrix
    #  ------------------------
    #  |                      |
    #  |a0-----a3a4-----a7....|
    #  |b0 B00 b3b4 B01 b7....|
    #  |c0     c3c4     c7....|
    #  |d0-----d3d4-----d7....|
    #  |e0-----e3e4-----e7....|
    #  |f0 B10 f3f4 B11 f7....|
    #  |g0     g3g4     g7....|
    #  |h0-----h3h4-----h7....|
    #  |                      |
    #  |                      |
    #  ------------------------
    #  {v0.2d[1], v0.2d[0]} = B00[0]+ B01[0] + B10[0] + B11[0]
    #  {v1.2d[1], v1.2d[0]} = B00[1]+ B01[1] + B10[1] + B11[1]
    #  {v2.2d[1], v2.2d[0]} = B00[2]+ B01[2] + B10[2] + B11[2]
    #  {v3.2d[1], v3.2d[0]} = B00[3]+ B01[3] + B10[3] + B11[3]
    #  v0 = e7 e6 e5 e4 e3 e2 e1 e0; a7 a6 a5 a4 a3 a2 a1 a0
    #  v1 = f7 f6 f5 f4 f3 f2 f1 f0; b7 b6 b5 b4 b3 b2 b1 b0
    #  v2 = g7 g6 g5 g4 g3 g2 g1 g0; c7 c6 c5 c4 c3 c2 c1 c0
    #  v3 = h7 h6 h5 h4 h3 h2 h1 h0; d7 d6 d5 d4 d3 d2 d1 d0
    #  Sequence:
    #  TRN1 v4.16b, v0.16b, v1.16b
    #  TRN2 v5.16b, v0.16b, v1.16b
    #  TRN1 v6.16b, v2.16b, v3.16b
    #  TRN2 v7.16b, v2.16b, v3.16b
    #  Now we have
    #  v4 = f6 e6 f4 e4 f2 e2 f0 e0; b6 a6 b4 a4 b2 a2 b0 a0
    #  v5 = f7 e7 f5 e5 f3 e3 f1 e1; b7 a7 b5 a5 b3 a3 b1 a1
    #  v6 = h6 g6 h4 g4 h2 g2 h0 g0; d6 c6 d4 c4 d2 c2 d0 c0
    #  v7 = h7 g7 h5 g5 h3 g3 h1 g1; d7 c7 d5 c5 d3 c3 d1 c1
    #  TRN1 v0.8h, v4.8h, v6.8h
    #  TRN2 v2.8h, v4.8h, v6.8h
    #  TRN1 v1.8h, v5.8h, v7.8h
    #  TRN2 v3.8h, v5.8h, v7.8h
    #  v0 = h4 g4 f4 e4 h0 g0 f0 e0; d4 c4 b4 a4 d0 c0 b0 a0
    #  v1 = h5 g5 f5 e5 h1 g1 f1 e1; d5 c5 b5 a5 d1 c1 b1 a1
    #  v2 = h6 g6 f6 e6 h2 g2 f2 e2; d6 c6 b6 a6 d2 c2 b2 a2
    #  v3 = h7 g7 f7 e7 h3 g3 f3 e3; d7 c7 b7 a7 d3 c3 b3 a3
    #  UZP1 v4.4s, v0.4s, v1.4s
    #  UZP2 v6.4s, v0.4s, v1.4s
    #  UZP1 v5.4s, v2.4s, v3.4s
    #  UZP2 v7.4s, v2.4s, v3.4s
    #  v4 = h1 g1 f1 e1 d1 c1 b1 a1; h0 g0 f0 e0 d0 c0 b0 a0
    #  v5 = h3 g3 f3 e3 d3 c3 b3 a3; h2 g2 f2 e2 d2 c2 b2 a2
    #  v6 = h5 g5 f5 e5 d5 c5 b5 a5; h4 g4 f4 e4 d4 c4 b4 a4
    #  v7 = h7 g7 f7 e7 d7 c7 b7 a7; h6 g6 f6 e6 d6 c6 b6 a6
    #  Thus 2 8x4 blocks are transposed.

    TRN1 v4.16b, v0.16b, v1.16b
    TRN2 v5.16b, v0.16b, v1.16b
    TRN1 v6.16b, v2.16b, v3.16b
    TRN2 v7.16b, v2.16b, v3.16b

    TRN1 v0.8h, v4.8h, v6.8h
    TRN2 v2.8h, v4.8h, v6.8h
    TRN1 v1.8h, v5.8h, v7.8h
    TRN2 v3.8h, v5.8h, v7.8h

    UZP1 v4.4s, v0.4s, v1.4s
    UZP2 v6.4s, v0.4s, v1.4s
    UZP1 v5.4s, v2.4s, v3.4s
    UZP2 v7.4s, v2.4s, v3.4s

    ST1 {v4.16b}, [x4], 16
    ST1 {v5.16b}, [x4], 16
    ST1 {v6.16b}, [x4], 16
    ST1 {v7.16b}, [x4], 16

    SUBS x1, x1, 2

    B.HS k_loop
1:
    CMP x1, -2
    B.EQ 2f

    LD1 {v0.s}[0], [x2]
    LD1 {v0.s}[1], [x8]
    LD1 {v1.s}[0], [x5]
    LD1 {v1.s}[1], [x9]
    LD1 {v2.s}[0], [x6]
    LD1 {v2.s}[1], [x10]
    LD1 {v3.s}[0], [x7]
    LD1 {v3.s}[1], [x11]

    #  Now we have 8x4 block of values that we will tranpose
    #  A matrix
    #  ----------------------------
    #  |                          |
    #  |                 a0-----a3|
    #  |                 b0 B00 b3|
    #  |   last block    c0     c3|
    #  |                 d0-----d3|
    #  |                 e0-----e3|
    #  |                 f0 B01 f3|
    #  |                 g0     g3|
    #  |                 h0-----h3|
    #  |                          |
    #  |                          |
    #  ---------------------------
    #  v0 = -; e3 e2 e1 e0 a3 a2 a1 a0
    #  v1 = -; f3 f2 f1 f0 b3 b2 b1 b0
    #  v2 = -; g3 g2 g1 g0 c3 c2 c1 c0
    #  v3 = -; h3 h2 h1 h0 d3 d2 d1 d0
    #  Sequence:
    #  TRN1 v4.16b, v0.16b, v1.16b
    #  TRN2 v5.16b, v0.16b, v1.16b
    #  TRN1 v6.16b, v2.16b, v3.16b
    #  TRN2 v7.16b, v2.16b, v3.16b
    #  Now we have
    #  v4 = -;f2 e2 f0 e0 b2 a2 b0 a0
    #  v5 = -;f3 e3 f1 e1 b3 a3 b1 a1
    #  v6 = -;h2 g2 h0 g0 d2 c2 d0 c0
    #  v7 = -;h3 g3 h1 g1 d3 c3 d1 c1
    #  TRN1 v0.8h, v4.8h, v6.8h
    #  TRN2 v2.8h, v4.8h, v6.8h
    #  TRN1 v1.8h, v5.8h, v7.8h
    #  TRN2 v3.8h, v5.8h, v7.8h
    #  v0 = -;h0 g0 f0 e0 d0 c0 b0 a0
    #  v1 = -;h1 g1 f1 e1 d1 c1 b1 a1
    #  v2 = -;h2 g2 f2 e2 d2 c2 b2 a2
    #  v3 = -;h3 g3 f3 e3 d3 c3 b3 a3
    #  Thus 1 8x4 blocks are transposed.

    TRN1 v4.16b, v0.16b, v1.16b
    TRN2 v5.16b, v0.16b, v1.16b
    TRN1 v6.16b, v2.16b, v3.16b
    TRN2 v7.16b, v2.16b, v3.16b
    TRN1 v0.8h, v4.8h, v6.8h
    TRN2 v2.8h, v4.8h, v6.8h
    TRN1 v1.8h, v5.8h, v7.8h
    TRN2 v3.8h, v5.8h, v7.8h

    ST1 {v0.8b}, [x4], 8
    ST1 {v1.8b}, [x4], 8
    ST1 {v2.8b}, [x4], 8
    ST1 {v3.8b}, [x4]
    .p2align 4
2:
    RET

END_FUNCTION pytorch_q8gemm_sparse_packA_ukernel_8x4__aarch64_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
