/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#ifdef __ELF__
    .macro BEGIN_FUNCTION name
        .text
        .align 2
        .global \name
        .type \name, %function
        \name:
    .endm

    .macro END_FUNCTION name
        .size \name, .-\name
    .endm
#elif defined(__MACH__)
    .macro BEGIN_FUNCTION name
        .text
        .align 2
        .global _\name
        .private_extern _\name
        _\name:
    .endm

    .macro END_FUNCTION name
    .endm
#endif
