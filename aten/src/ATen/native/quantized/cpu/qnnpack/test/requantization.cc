/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstddef>
#include <cstdlib>

#include <cpuinfo.h>
#include <gtest/gtest.h>
#include <qnnpack/requantization-stubs.h>

#include "requantization-tester.h"

/*
 * Precise scalar implementation using unsigned 32-bit arithmetics.
 */

TEST(PRECISE__SCALAR_UNSIGNED32, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_precise__scalar_unsigned32);
  }
}

TEST(PRECISE__SCALAR_UNSIGNED32, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_precise__scalar_unsigned32);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED32, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(
              pytorch_qnnp_requantize_precise__scalar_unsigned32);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED32, divide_by_po2_with_rounding_down) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingDown(
              pytorch_qnnp_requantize_precise__scalar_unsigned32);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED32, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(
              pytorch_qnnp_requantize_precise__scalar_unsigned32);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED32, special_cases) {
  RequantizationTester().testSpecialCases(
      pytorch_qnnp_requantize_precise__scalar_unsigned32);
}

TEST(PRECISE__SCALAR_UNSIGNED32, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesPrecise(
      pytorch_qnnp_requantize_precise__scalar_unsigned32);
}

/*
 * Precise scalar implementation using unsigned 64-bit arithmetics.
 */

TEST(PRECISE__SCALAR_UNSIGNED64, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_precise__scalar_unsigned64);
  }
}

TEST(PRECISE__SCALAR_UNSIGNED64, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_precise__scalar_unsigned64);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED64, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(
              pytorch_qnnp_requantize_precise__scalar_unsigned64);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED64, divide_by_po2_with_rounding_down) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingDown(
              pytorch_qnnp_requantize_precise__scalar_unsigned64);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED64, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(
              pytorch_qnnp_requantize_precise__scalar_unsigned64);
    }
  }
}

TEST(PRECISE__SCALAR_UNSIGNED64, special_cases) {
  RequantizationTester().testSpecialCases(
      pytorch_qnnp_requantize_precise__scalar_unsigned64);
}

TEST(PRECISE__SCALAR_UNSIGNED64, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesPrecise(
      pytorch_qnnp_requantize_precise__scalar_unsigned64);
}

/*
 * Precise scalar implementation using signed 64-bit arithmetics.
 */

TEST(PRECISE__SCALAR_SIGNED64, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_precise__scalar_signed64);
  }
}

TEST(PRECISE__SCALAR_SIGNED64, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_precise__scalar_signed64);
    }
  }
}

TEST(PRECISE__SCALAR_SIGNED64, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(
              pytorch_qnnp_requantize_precise__scalar_signed64);
    }
  }
}

TEST(PRECISE__SCALAR_SIGNED64, divide_by_po2_with_rounding_down) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingDown(
              pytorch_qnnp_requantize_precise__scalar_signed64);
    }
  }
}

TEST(PRECISE__SCALAR_SIGNED64, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(
              pytorch_qnnp_requantize_precise__scalar_signed64);
    }
  }
}

TEST(PRECISE__SCALAR_SIGNED64, special_cases) {
  RequantizationTester().testSpecialCases(
      pytorch_qnnp_requantize_precise__scalar_signed64);
}

TEST(PRECISE__SCALAR_SIGNED64, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesPrecise(
      pytorch_qnnp_requantize_precise__scalar_signed64);
}

/*
 * FP32-based scalar implementation using lrintf function.
 */

TEST(FP32__SCALAR_LRINTF, random_cases) {
  RequantizationTester().iterations(1000).testRandomCasesApproximate(
      pytorch_qnnp_requantize_fp32__scalar_lrintf);
}

/*
 * FP32-based scalar implementation using magic trick for FP32->INT32
 * conversion.
 */

TEST(FP32__SCALAR_MAGIC, random_cases) {
  RequantizationTester().iterations(1000).testRandomCasesApproximate(
      pytorch_qnnp_requantize_fp32__scalar_magic);
}

/*
 * Q31-based scalar implementation.
 */

TEST(Q31__SCALAR, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_q31__scalar);
  }
}

TEST(Q31__SCALAR, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_q31__scalar);
    }
  }
}

TEST(Q31__SCALAR, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_q31__scalar);
    }
  }
}

/* No rounding down test - it fails because of upward bias in multiplication */

TEST(Q31__SCALAR, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(pytorch_qnnp_requantize_q31__scalar);
    }
  }
}

TEST(Q31__SCALAR, special_cases) {
  RequantizationTester().testSpecialCases(pytorch_qnnp_requantize_q31__scalar);
}

TEST(Q31__SCALAR, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesApproximate(
      pytorch_qnnp_requantize_q31__scalar);
}

TEST(Q31__SCALAR, random_match_gemmlowp) {
  RequantizationTester().iterations(100).testRandomCasesAgainstReference(
      pytorch_qnnp_requantize_q31__scalar,
      pytorch_qnnp_requantize_gemmlowp__scalar);
}

/*
 * Scalar implementation from gemmlowp.
 */

TEST(GEMMLOWP__SCALAR, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesApproximate(
      pytorch_qnnp_requantize_gemmlowp__scalar);
}

/*
 * Precise PSIMD implementation using unsigned 32-bit arithmetics.
 */

TEST(PRECISE__PSIMD, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_precise__psimd);
  }
}

TEST(PRECISE__PSIMD, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_precise__psimd);
    }
  }
}

TEST(PRECISE__PSIMD, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(
              pytorch_qnnp_requantize_precise__psimd);
    }
  }
}

TEST(PRECISE__PSIMD, divide_by_po2_with_rounding_down) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingDown(
              pytorch_qnnp_requantize_precise__psimd);
    }
  }
}

TEST(PRECISE__PSIMD, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(
              pytorch_qnnp_requantize_precise__psimd);
    }
  }
}

TEST(PRECISE__PSIMD, special_cases) {
  RequantizationTester().testSpecialCases(
      pytorch_qnnp_requantize_precise__psimd);
}

TEST(PRECISE__PSIMD, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesPrecise(
      pytorch_qnnp_requantize_precise__psimd);
}

/*
 * FP32-based PSIMD implementation using magic trick for FP32->INT32 conversion.
 */

TEST(FP32__PSIMD, random_cases) {
  RequantizationTester().iterations(1000).testRandomCasesApproximate(
      pytorch_qnnp_requantize_fp32__psimd);
}

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64

/*
 * Precise SSE2 implementation using floating-point shuffle.
 */

TEST(PRECISE__SSE2, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_precise__sse2);
  }
}

TEST(PRECISE__SSE2, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_precise__sse2);
    }
  }
}

TEST(PRECISE__SSE2, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_precise__sse2);
    }
  }
}

TEST(PRECISE__SSE2, divide_by_po2_with_rounding_down) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingDown(
              pytorch_qnnp_requantize_precise__sse2);
    }
  }
}

TEST(PRECISE__SSE2, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(
              pytorch_qnnp_requantize_precise__sse2);
    }
  }
}

TEST(PRECISE__SSE2, special_cases) {
  RequantizationTester().testSpecialCases(
      pytorch_qnnp_requantize_precise__sse2);
}

TEST(PRECISE__SSE2, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesPrecise(
      pytorch_qnnp_requantize_precise__sse2);
}

/*
 * Precise SSSE3 implementation using floating-point shuffle.
 */

TEST(PRECISE__SSSE3, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_precise__ssse3);
  }
}

TEST(PRECISE__SSSE3, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_precise__ssse3);
    }
  }
}

TEST(PRECISE__SSSE3, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(
              pytorch_qnnp_requantize_precise__ssse3);
    }
  }
}

TEST(PRECISE__SSSE3, divide_by_po2_with_rounding_down) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingDown(
              pytorch_qnnp_requantize_precise__ssse3);
    }
  }
}

TEST(PRECISE__SSSE3, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(
              pytorch_qnnp_requantize_precise__ssse3);
    }
  }
}

TEST(PRECISE__SSSE3, special_cases) {
  RequantizationTester().testSpecialCases(
      pytorch_qnnp_requantize_precise__ssse3);
}

TEST(PRECISE__SSSE3, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesPrecise(
      pytorch_qnnp_requantize_precise__ssse3);
}

/*
 * Precise SSE4.1 implementation using static blend instruction.
 */

TEST(PRECISE__SSE4, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_precise__sse4);
  }
}

TEST(PRECISE__SSE4, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_precise__sse4);
    }
  }
}

TEST(PRECISE__SSE4, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_precise__sse4);
    }
  }
}

TEST(PRECISE__SSE4, divide_by_po2_with_rounding_down) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingDown(
              pytorch_qnnp_requantize_precise__sse4);
    }
  }
}

TEST(PRECISE__SSE4, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(
              pytorch_qnnp_requantize_precise__sse4);
    }
  }
}

TEST(PRECISE__SSE4, special_cases) {
  RequantizationTester().testSpecialCases(
      pytorch_qnnp_requantize_precise__sse4);
}

TEST(PRECISE__SSE4, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesPrecise(
      pytorch_qnnp_requantize_precise__sse4);
}

/*
 * FP32-based x86 SSE2 implementation.
 */

TEST(FP32__SSE2, random_cases) {
  RequantizationTester().iterations(1000).testRandomCasesApproximate(
      pytorch_qnnp_requantize_fp32__sse2);
}

/*
 * Q31-based x86 SSE2 implementation.
 */

TEST(Q31__SSE2, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_q31__sse2);
  }
}

TEST(Q31__SSE2, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_q31__sse2);
    }
  }
}

TEST(Q31__SSE2, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_q31__sse2);
    }
  }
}

/* No rounding down test - it fails because of upward bias in multiplication */

TEST(Q31__SSE2, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(pytorch_qnnp_requantize_q31__sse2);
    }
  }
}

TEST(Q31__SSE2, special_cases) {
  RequantizationTester().testSpecialCases(pytorch_qnnp_requantize_q31__sse2);
}

TEST(Q31__SSE2, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesApproximate(
      pytorch_qnnp_requantize_q31__sse2);
}

TEST(Q31__SSE2, random_match_gemmlowp) {
  RequantizationTester().iterations(100).testRandomCasesAgainstReference(
      pytorch_qnnp_requantize_q31__sse2,
      pytorch_qnnp_requantize_gemmlowp__sse2);
}

/*
 * Q31-based x86 SSSE3 implementation.
 */

TEST(Q31__SSSE3, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_q31__ssse3);
  }
}

TEST(Q31__SSSE3, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_q31__ssse3);
    }
  }
}

TEST(Q31__SSSE3, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_q31__ssse3);
    }
  }
}

/* No rounding down test - it fails because of upward bias in multiplication */

TEST(Q31__SSSE3, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(pytorch_qnnp_requantize_q31__ssse3);
    }
  }
}

TEST(Q31__SSSE3, special_cases) {
  RequantizationTester().testSpecialCases(pytorch_qnnp_requantize_q31__ssse3);
}

TEST(Q31__SSSE3, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesApproximate(
      pytorch_qnnp_requantize_q31__ssse3);
}

TEST(Q31__SSSE3, random_match_gemmlowp) {
  RequantizationTester().iterations(100).testRandomCasesAgainstReference(
      pytorch_qnnp_requantize_q31__ssse3,
      pytorch_qnnp_requantize_gemmlowp__ssse3);
}

/*
 * Q31-based x86 SSE4 implementation.
 */

TEST(Q31__SSE4, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_q31__sse4);
  }
}

TEST(Q31__SSE4, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_q31__sse4);
    }
  }
}

TEST(Q31__SSE4, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_q31__sse4);
    }
  }
}

/* No rounding down test - it fails because of upward bias in multiplication */

TEST(Q31__SSE4, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(pytorch_qnnp_requantize_q31__sse4);
    }
  }
}

TEST(Q31__SSE4, special_cases) {
  RequantizationTester().testSpecialCases(pytorch_qnnp_requantize_q31__sse4);
}

TEST(Q31__SSE4, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesApproximate(
      pytorch_qnnp_requantize_q31__sse4);
}

TEST(Q31__SSE4, random_match_gemmlowp) {
  RequantizationTester().iterations(100).testRandomCasesAgainstReference(
      pytorch_qnnp_requantize_q31__sse4,
      pytorch_qnnp_requantize_gemmlowp__sse4);
}

/*
 * x86 SSE2 implementation from gemmlowp.
 */

TEST(GEMMLOWP__SSE2, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_gemmlowp__sse2);
  }
}

TEST(GEMMLOWP__SSE2, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_gemmlowp__sse2);
    }
  }
}

TEST(GEMMLOWP__SSE2, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(
              pytorch_qnnp_requantize_gemmlowp__sse2);
    }
  }
}

/* No rounding down test - it fails because of upward bias in multiplication */

TEST(GEMMLOWP__SSE2, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(
              pytorch_qnnp_requantize_gemmlowp__sse2);
    }
  }
}

TEST(GEMMLOWP__SSE2, special_cases) {
  RequantizationTester().testSpecialCases(
      pytorch_qnnp_requantize_gemmlowp__sse2);
}

TEST(GEMMLOWP__SSE2, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesApproximate(
      pytorch_qnnp_requantize_gemmlowp__sse2);
}

/*
 * x86 SSSE3 implementation from gemmlowp.
 */

TEST(GEMMLOWP__SSSE3, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_gemmlowp__ssse3);
  }
}

TEST(GEMMLOWP__SSSE3, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_gemmlowp__ssse3);
    }
  }
}

TEST(GEMMLOWP__SSSE3, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(
              pytorch_qnnp_requantize_gemmlowp__ssse3);
    }
  }
}

/* No rounding down test - it fails because of upward bias in multiplication */

TEST(GEMMLOWP__SSSE3, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(
              pytorch_qnnp_requantize_gemmlowp__ssse3);
    }
  }
}

TEST(GEMMLOWP__SSSE3, special_cases) {
  RequantizationTester().testSpecialCases(
      pytorch_qnnp_requantize_gemmlowp__ssse3);
}

TEST(GEMMLOWP__SSSE3, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesApproximate(
      pytorch_qnnp_requantize_gemmlowp__ssse3);
}

/*
 * x86 SSE4 implementation from gemmlowp.
 */

TEST(GEMMLOWP__SSE4, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_gemmlowp__sse4);
  }
}

TEST(GEMMLOWP__SSE4, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_gemmlowp__sse4);
    }
  }
}

TEST(GEMMLOWP__SSE4, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(
              pytorch_qnnp_requantize_gemmlowp__sse4);
    }
  }
}

/* No rounding down test - it fails because of upward bias in multiplication */

TEST(GEMMLOWP__SSE4, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(
              pytorch_qnnp_requantize_gemmlowp__sse4);
    }
  }
}

TEST(GEMMLOWP__SSE4, special_cases) {
  RequantizationTester().testSpecialCases(
      pytorch_qnnp_requantize_gemmlowp__sse4);
}

TEST(GEMMLOWP__SSE4, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesApproximate(
      pytorch_qnnp_requantize_gemmlowp__sse4);
}

#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64

/*
 * Precise ARM NEON implementation.
 */

TEST(PRECISE__NEON, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_precise__neon);
  }
}

TEST(PRECISE__NEON, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_precise__neon);
    }
  }
}

TEST(PRECISE__NEON, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_precise__neon);
    }
  }
}

TEST(PRECISE__NEON, divide_by_po2_with_rounding_down) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingDown(
              pytorch_qnnp_requantize_precise__neon);
    }
  }
}

TEST(PRECISE__NEON, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(
              pytorch_qnnp_requantize_precise__neon);
    }
  }
}

TEST(PRECISE__NEON, special_cases) {
  RequantizationTester().testSpecialCases(
      pytorch_qnnp_requantize_precise__neon);
}

TEST(PRECISE__NEON, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesPrecise(
      pytorch_qnnp_requantize_precise__neon);
}

/*
 * FP32-based ARM NEON implementation.
 */

TEST(FP32__NEON, random_cases) {
  RequantizationTester().iterations(1000).testRandomCasesApproximate(
      pytorch_qnnp_requantize_fp32__neon);
}

/*
 * Q31-based ARM NEON implementation.
 */

TEST(Q31__NEON, exact_divide_by_po2) {
  for (uint32_t s = 1; s < 32; s++) {
    RequantizationTester().s(s).testExactDivideByPO2(
        pytorch_qnnp_requantize_q31__neon);
  }
}

TEST(Q31__NEON, exact_divide_by_po2_with_zero_point) {
  for (int32_t zeroPoint = 1; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester().zeroPoint(zeroPoint).s(s).testExactDivideByPO2(
          pytorch_qnnp_requantize_q31__neon);
    }
  }
}

TEST(Q31__NEON, divide_by_po2_with_rounding_up) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingUp(pytorch_qnnp_requantize_q31__neon);
    }
  }
}

/* No rounding down test - it fails because of upward bias in multiplication */

TEST(Q31__NEON, divide_by_po2_with_rounding_away) {
  for (int32_t zeroPoint = 0; zeroPoint < 256; zeroPoint++) {
    for (uint32_t s = 1; s < 32; s++) {
      RequantizationTester()
          .zeroPoint(zeroPoint)
          .s(s)
          .testDivideByPO2WithRoundingAway(pytorch_qnnp_requantize_q31__neon);
    }
  }
}

TEST(Q31__NEON, special_cases) {
  RequantizationTester().testSpecialCases(pytorch_qnnp_requantize_q31__neon);
}

TEST(Q31__NEON, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesApproximate(
      pytorch_qnnp_requantize_q31__neon);
}

TEST(Q31__NEON, random_match_gemmlowp) {
  RequantizationTester().iterations(100).testRandomCasesAgainstReference(
      pytorch_qnnp_requantize_q31__neon,
      pytorch_qnnp_requantize_gemmlowp__neon);
}

/*
 * ARM NEON implementation from gemmlowp.
 */

TEST(GEMMLOWP__NEON, random_cases) {
  RequantizationTester().iterations(100).testRandomCasesApproximate(
      pytorch_qnnp_requantize_gemmlowp__neon);
}

#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
