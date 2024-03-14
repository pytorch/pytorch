#include <immintrin.h>
#include "fp16_fma.h"

namespace fp16_fma {

typedef int int16;
typedef char int8;
typedef unsigned short int bits16;
typedef unsigned int bits32;
typedef signed char Word8;
typedef unsigned char UWord8;
typedef signed short Word16;
typedef unsigned short UWord16;
typedef signed int Word32;
typedef unsigned int UWord32;
typedef long long Word64;
typedef unsigned long long UWord64;
typedef unsigned short float16;
typedef signed int sbits32;
typedef signed short int sbits16;

typedef char flag;

#define MAX_U32 (UWord32)0xffffffffL
#define MAX_U16 (UWord16)0xffff
#define BITMASK_T(typ, w) (((typ)1 << (w)) - 1)
#define TESTBIT(x, n) (((x) >> (n)) & 1)

#define float16_default_nan 0x7E00
#define float16_default_nan_pos 0x7E00
#define float16_default_nan_neg 0xFE00

int8 float_exception_flags = 0;

enum {
  float_round_nearest_even = 0,
  float_round_down = 1,
  float_round_up = 2,
  float_round_to_zero = 3
};

int8 float_rounding_mode = float_round_nearest_even;
enum { float_tininess_after_rounding = 0, float_tininess_before_rounding = 1 };
int float_detect_tininess = float_tininess_after_rounding;

inline bits16 extractFloat16Frac(float16 a) {
  return a & 0x3FF;
}

inline int16 extractFloat16Exp(float16 a) {
  return (a >> 10) & 0x1F;
}

inline flag extractFloat16Sign(float16 a) {
  return a >> 15;
}

flag float16_is_quiet_nan(float16 a) {
  return (0xFC00 <= (bits16)(a << 1));
}

flag float16_is_signaling_nan(float16 a) {
  return (((a >> 9) & 0x3F) == 0x3E) && (a & 0x01FF);
}

enum {
  float_flag_inexact = 1,
  float_flag_divbyzero = 2,
  float_flag_underflow = 4,
  float_flag_overflow = 8,
  float_flag_invalid = 16
};

void float_raise(int8 flags) {
  float_exception_flags |= flags;
}
int pickNaNMulAdd(
    flag aIsQNaN,
    flag aIsSNaN,
    flag bIsQNaN,
    flag bIsSNaN,
    flag cIsQNaN,
    flag cIsSNaN,
    flag infzero) {
  if (infzero) {
    float_raise(float_flag_invalid);
    return 2;
  }

  if (cIsSNaN || cIsQNaN) {
    return 2;
  } else if (bIsSNaN || bIsQNaN) {
    return 1;
  } else {
    return 0;
  }
}

inline float16 packFloat16(flag zSign, int16 zExp, bits16 zSig) {
  return (((bits16)zSign) << 15) + (((bits16)zExp) << 10) + zSig;
}

float16
propagateFloat16MulAddNaN(float16 a, float16 b, float16 c, flag infzero) {
  flag aIsQuietNaN, aIsSignalingNaN, bIsQuietNaN, bIsSignalingNaN, cIsQuietNaN,
      cIsSignalingNaN;
  int selNaN;

  aIsQuietNaN = float16_is_quiet_nan(a);
  aIsSignalingNaN = float16_is_signaling_nan(a);
  bIsQuietNaN = float16_is_quiet_nan(b);
  bIsSignalingNaN = float16_is_signaling_nan(b);
  cIsQuietNaN = float16_is_quiet_nan(c);
  cIsSignalingNaN = float16_is_signaling_nan(c);

  if (aIsSignalingNaN | bIsSignalingNaN | cIsSignalingNaN) {
    float_raise(float_flag_invalid);
  }

  selNaN = pickNaNMulAdd(
      aIsQuietNaN,
      aIsSignalingNaN,
      bIsQuietNaN,
      bIsSignalingNaN,
      cIsQuietNaN,
      cIsSignalingNaN,
      infzero);

  switch (selNaN) {
    case 0:
      return a | (1 << 9);
    case 1:
      return b | (1 << 9);
    case 2:
      return c | (1 << 9);
    case 3:
    default:
      return float16_default_nan;
  }
}

inline void shift32RightJamming(bits32 a, int16 count, bits32* zPtr) {
  bits32 z;

  if (count == 0) {
    z = a;
  } else if (count < 32) {
    z = (a >> count) | ((a << ((-count) & 31)) != 0);
  } else {
    z = (a != 0);
  }
  *zPtr = z;
}

void shift16RightJamming(bits16 a, int16 count, bits16* zPtr) {
  bits16 z;

  if (count == 0) {
    z = a;
  } else if (count < 16) {
    z = (a >> count) | (((a << ((-count) & 15)) & 0xffff) != 0);
  } else {
    z = (a != 0);
  }
  *zPtr = z;
}

Word8 GetRound(Word32 fcr) {
  Word8 res, round_mode;
  round_mode = fcr & 0x3; // lower 2 bits as rounding mode in FCR
  res = (round_mode == 3)
      ? 1
      : ((round_mode == 2)
             ? 2
             : ((round_mode == 1) ? 3 : 0)); // Translate to float_rounding_mode
  return res;
}

Word8 GetException(Word32 fsr) {
  Word8 res = 0;
  if (TESTBIT(fsr, 7) == 1)
    res |= 32; // float_flag_inexact
  if (TESTBIT(fsr, 8) == 1)
    res |= 16; // float_flag_underflow
  if (TESTBIT(fsr, 9) == 1)
    res |= 8; // float_flag_overflow
  if (TESTBIT(fsr, 10) == 1)
    res |= 4; // float_flag_divbyzero
  if (TESTBIT(fsr, 11) == 1)
    res |= 1; // float_flag_invalid
  return res;
}

float16 roundAndPackFloat16(flag zSign, int16 zExp, bits16 zSig) {
  int8 roundingMode;
  flag roundNearestEven;
  int8 roundIncrement, roundBits;
  flag isTiny;

  roundingMode = float_rounding_mode;
  roundNearestEven = (roundingMode == float_round_nearest_even);
  roundIncrement = 0x8;
  if (!roundNearestEven) {
    //    if ( ( ! roundNearestEven ) && ( roundingMode !=
    //    float_round_ties_away) ) {
    if (roundingMode == float_round_to_zero) {
      roundIncrement = 0;
    } else {
      roundIncrement = 0xF;
      if (zSign) {
        if (roundingMode == float_round_up)
          roundIncrement = 0;
      } else {
        if (roundingMode == float_round_down)
          roundIncrement = 0;
      }
    }
  }
  roundBits = zSig & 0xF;
  if (0x1D <= (bits16)zExp) {
    if ((0x1D < zExp) ||
        ((zExp == 0x1D) && ((sbits16)(zSig + roundIncrement) < 0))) {
      float_raise(float_flag_overflow | float_flag_inexact);
      return packFloat16(zSign, 0x1F, 0) - (roundIncrement == 0);
    }
    if (zExp < 0) {
      isTiny = (float_detect_tininess == float_tininess_before_rounding) ||
          (zExp < -1) || (zSig + roundIncrement < 0x8000);
      shift16RightJamming(zSig, -zExp, &zSig);
      zExp = 0;
      roundBits = zSig & 0xF;

      if (isTiny && roundBits)
        float_raise(float_flag_underflow);
    }
  }
  if (roundBits)
    float_exception_flags |= float_flag_inexact;
  zSig = (zSig + roundIncrement) >> 4;
  zSig &= ~(((roundBits ^ 0x8) == 0) & roundNearestEven);
  if (zSig == 0)
    zExp = 0;
  return packFloat16(zSign, zExp, zSig);
}

int8 countLeadingZeros32(bits32 a) {
  static const int8 countLeadingZerosHigh[] = {
      8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int8 shiftCount;

  shiftCount = 0;
  if (a < 0x10000) {
    shiftCount += 16;
    a <<= 16;
  }
  if (a < 0x1000000) {
    shiftCount += 8;
    a <<= 8;
  }
  shiftCount += countLeadingZerosHigh[a >> 24];
  return shiftCount;
}

void normalizeFloat16Subnormal(bits16 aSig, int16* zExpPtr, bits16* zSigPtr) {
  int8 shiftCount;

  shiftCount = countLeadingZeros32((bits32)aSig) - 16 - 5;
  *zSigPtr = aSig << shiftCount;
  *zExpPtr = 1 - shiftCount;
}

float16 float16_muladd(float16 a, float16 b, float16 c, flag negate_product) {
  flag aSign, bSign, cSign, zSign;
  int16 aExp, bExp, cExp, pExp, zExp, expDiff;
  bits16 aSig, bSig, cSig;
  flag pInf, pZero, pSign;
  bits32 pSig32, cSig32, zSig32;
  bits16 pSig;
  int shiftcount;
  flag infzero;

  /* Extract the sign bit, exponent and significant  */
  aSig = extractFloat16Frac(a);
  aExp = extractFloat16Exp(a);
  aSign = extractFloat16Sign(a);

  bSig = extractFloat16Frac(b);
  bExp = extractFloat16Exp(b);
  bSign = extractFloat16Sign(b);

  cSig = extractFloat16Frac(c);
  cExp = extractFloat16Exp(c);
  cSign = extractFloat16Sign(c);

  /* Flag to indicate fusedMultiplyAdd(0, inf,  or fusedMultiplyAdd(inf, 0 c) */
  infzero =
      ((aExp == 0 && aSig == 0 && bExp == 0x1f && bSig == 0) ||
       (aExp == 0x1f && aSig == 0 && bExp == 0 && bSig == 0));

  /* CASE1: if any input is NaN =>  NaN propagate */

  /* It is implementation-defined whether the cases of (0,inf,qnan)
   * and (inf,0,qnan) raise InvalidOperation or not (and what QNaN
   * they return if they do), so we have to hand this information
   * off to the target-specific pick-a-NaN routine.
   */

  /* IEEE754 7.2 - Invalid: fusedMultiplyAdd(0, inf, c) or
   * fusedMultiplyAdd(inf, 0 , c) unless c is a quiet NaN; If c is a
   * quiet NaN then it is implementation defined whether the invalid operation
   * exception is signaled.
   */
  if (((aExp == 0x1f) && aSig) || ((bExp == 0x1f) && bSig) ||
      ((cExp == 0x1f) && cSig)) {
    return propagateFloat16MulAddNaN(a, b, c, infzero);
  }

  /* Work out the sign and type of the product */
  pSign = aSign ^ bSign;
  if (negate_product) {
    pSign ^= 1;
  }

  /* CASE2: fusedMultiplyAdd(0, inf, c) or fusedMultiplyAdd(inf,0,  c) and c is
   * not NaN  => raise invalid */
  if (infzero) {
    float_raise(float_flag_invalid);
    return float16_default_nan;
  }

  pInf = (aExp == 0x1f) || (bExp == 0x1f);
  pZero = ((aExp | aSig) == 0) || ((bExp | bSig) == 0);

  /* CASE3 and CASE4: c is inf, p is number or inf*/
  if (cExp == 0x1f) {
    if (pInf && (pSign ^ cSign)) {
      /* CASE3: addition of opposite-signed infinities => InvalidOperation */
      float_raise(float_flag_invalid);
      return float16_default_nan;
    }
    /* CASE4: Otherwise generate an infinity of the same sign */
    return packFloat16(cSign, 0x1f, 0);
  }

  /* CASE5: c is number and p is inf */
  if (pInf) {
    return packFloat16(pSign, 0x1f, 0);
  }

  /* CASE6: c is number, p is zero */
  if (pZero) {
    if (cExp == 0) {
      if (cSig == 0) {
        /* Adding two exact zeroes */
        if (pSign == cSign) {
          zSign = pSign;
        } else if (float_rounding_mode == float_round_down) {
          zSign = 1;
        } else {
          zSign = 0;
        }
        return packFloat16(zSign, 0, 0);
      }
    }
    /* CASE7: Zero plus something non-zero : just return the something */
    return c;
  }

  if (aExp == 0) {
    normalizeFloat16Subnormal(aSig, &aExp, &aSig);
  }
  if (bExp == 0) {
    normalizeFloat16Subnormal(bSig, &bExp, &bSig);
  }

  /* Calculate the actual result a * b + c */

  /* NOTE: we subtract 0x7e where float16_mul() subtracts 0x7f
   * because we want the true exponent, not the "one-less-than"
   * flavour that roundAndPackFloat16() takes.
   */
  pExp = aExp + bExp - 0xe;
  aSig = (aSig | 0x0400) << 4;
  bSig = (bSig | 0x0400) << 5;
  pSig32 = (bits32)aSig * bSig;
  if ((sbits32)(pSig32 << 1) >= 0) {
    pSig32 <<= 1;
    pExp--;
  }

  zSign = pSign;

  /* Now pSig32 is the significand of the multiply, with the explicit bit in
   * position 30.
   */
  if (cExp == 0) {
    if (!cSig) {
      /* Throw out the special case of c being an exact zero now */
      shift32RightJamming(pSig32, 16, &pSig32);
      pSig = pSig32;
      return roundAndPackFloat16(zSign, pExp - 1, pSig);
    }
    normalizeFloat16Subnormal(cSig, &cExp, &cSig);
  }

  cSig32 = (bits32)cSig << (30 - 10);
  cSig32 |= 0x40000000;
  expDiff = pExp - cExp;

  if (pSign == cSign) {
    /* Addition */
    if (expDiff > 0) {
      /* scale c to match p */
      shift32RightJamming(cSig32, expDiff, &cSig32);
      zExp = pExp;
    } else if (expDiff < 0) {
      /* scale p to match c */
      shift32RightJamming(pSig32, -expDiff, &pSig32);
      zExp = cExp;
    } else {
      /* no scaling needed */
      zExp = cExp;
    }
    /* Add significands and make sure explicit bit ends up in posn 62 */
    zSig32 = pSig32 + cSig32;
    if ((sbits32)zSig32 < 0) {
      shift32RightJamming(zSig32, 1, &zSig32);
    } else {
      zExp--;
    }
  } else {
    /* Subtraction */
    if (expDiff > 0) {
      shift32RightJamming(cSig32, expDiff, &cSig32);
      zSig32 = pSig32 - cSig32;
      zExp = pExp;
    } else if (expDiff < 0) {
      shift32RightJamming(pSig32, -expDiff, &pSig32);
      zSig32 = cSig32 - pSig32;
      zExp = cExp;
      zSign ^= 1;
    } else {
      zExp = pExp;
      if (cSig32 < pSig32) {
        zSig32 = pSig32 - cSig32;
      } else if (pSig32 < cSig32) {
        zSig32 = cSig32 - pSig32;
        zSign ^= 1;
      } else {
        /* Exact zero */
        zSign = 0;
        if (float_rounding_mode == float_round_down) {
          zSign ^= 1;
        }
        return packFloat16(zSign, 0, 0);
      }
    }
    --zExp;
    /* Normalize to put the explicit bit back into bit 62. */
    shiftcount = countLeadingZeros32(zSig32) - 1;
    zSig32 <<= shiftcount;
    zExp -= shiftcount;
  }
  shift32RightJamming(zSig32, 16, &zSig32);
  return roundAndPackFloat16(zSign, zExp, zSig32);
}

void fp_mac_h(
    Word16 d0,
    Word16 d1,
    Word16 d2,
    Word32 negate_product,
    Word32 fcr,
    Word32 fsr_i,
    Word16* res,
    Word32* fsr_o) {
  // Extract rounding mode from FCR/FSR to softfloat
  float_rounding_mode = GetRound(fcr);
  float_exception_flags = GetException(fsr_i);
  // Call softfloat lib
  *res = float16_muladd(d1, d2, d0, negate_product);
  //*fsr_o =  PutException(float_exception_flags, fsr_i);
}

void fma16(
    const Word16 input,
    const Word16 a,
    const Word16 b,
    const Word32 fcr,
    const Word32 fsr_i,
    Word16* result,
    Word32* fsr_o) {
  Word16 res;
  Word32 fsr = 0;
  // Call fp utility
  fp_mac_h(b, input, a, 0, fcr, fsr_i, &res, &fsr);
  // Output result
  *fsr_o = fsr;
  *result = res;
}

float fake_fma_fp16_slow(float v1, float v2, float v3) {
  uint32_t fcr_val = 0;
  uint32_t fsr_val = 0x00000F80;
  uint32_t exception_flags = 0;

  uint16_t hv1, hv2, hv3, hresult;
  hv1 = _cvtss_sh(v1, 0);
  hv2 = _cvtss_sh(v2, 0);
  hv3 = _cvtss_sh(v3, 0);

  fma16(
      *reinterpret_cast<Word16*>(&hv1),
      *reinterpret_cast<Word16*>(&hv2),
      *reinterpret_cast<Word16*>(&hv3),
      *reinterpret_cast<Word32*>(&fcr_val),
      *reinterpret_cast<Word32*>(&fsr_val),
      reinterpret_cast<Word16*>(&hresult),
      reinterpret_cast<Word32*>(&exception_flags));

  return _cvtsh_ss(hresult);
}

void fake_fma_fp16_slow(int N, const float* A, const float* B, float* Out) {
  for (int n = 0; n < N; n++) {
    Out[n] = fake_fma_fp16_slow(A[n], B[n], Out[n]);
  }
}

} // namespace fp16_fma
