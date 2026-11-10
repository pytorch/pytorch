/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#ifdef __ELF__
#if defined(__aarch64__) && defined(__ARM_FEATURE_BTI_DEFAULT) && (__ARM_FEATURE_BTI_DEFAULT == 1)
    // BEGIN_FUNCTION starts every function with a BTI landing pad, so
    // advertise BTI for the object when branch-target enforcement is requested.
    .pushsection .note.gnu.property, "a", %note
    .balign 8
    .long 4
    .long 0x10
    .long 0x5
    .asciz "GNU"
    .long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
    .long 4
    .long 1 // GNU_PROPERTY_AARCH64_FEATURE_1_BTI
    .long 0
    .popsection
#define QNNPACK_BTI_C hint #34 // bti c
#else
#define QNNPACK_BTI_C
#endif

    .macro BEGIN_FUNCTION name
        .text
        .align 2
        .global \name
        .type \name, %function
        \name:
        QNNPACK_BTI_C
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
