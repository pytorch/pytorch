#include "fp16_fma.h"
#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace fake_fp16 {

// Compute fp16 FMA using fp16
// Out = FMA (A, B, Out)
//
// Algorithm:
//  Do an FMA in fp64
//  Since fp16 has 10 bits of mantissa and fp64 has 52, zero out
//   42 bits.
//  Extract the exponent.
//  If the exponent ends up in the subnormal range, shift out
//  only 42 - (14 + exponent).
//  Compute the bounce value as a value that is big enough to
//  push all the digits except for the required ones in fp16,
//  the objective is to push digits to let the machine do rounding.
//  Add 42 or the computed number (in case of denormals) to the exponent.
//  For negative numbers set the highest bit of the mantissa to 1.
void fma_fp16(int N, const float* A, const float* B, float* Out) {
  constexpr int blockSize = 4;
  constexpr uint64_t mask = 0x7ff0000000000000;
  constexpr uint64_t shift_bits = 52;
  constexpr uint64_t offset = 1023;
  constexpr uint64_t dbl_threehalf = 0x3ff8000000000000;

  uint64_t expo_bouncer;

  // It can be proven than in the absence of intermediate overflow
  // the desired numerical result can be obtained even with the
  // possibility of a double rounding, as follow.
  //    round-to-fp16-precision(   (double)A * (double)B + (double)C  )
  // This statement is not proved here; but we explain how to round a fp64
  // number into fp16 precision using the technique of a "Bouncer"
  // Suppose a numerical value in fp64 has exponent value of E
  // If -14 <= E <= 15 (the fp16 exponent value for normalized number),
  // the lsb of this value in fp16 precision is 2^(E-10).
  // Now consider this fp64 number Bouncer which is 2^(52+(E-10)) * 3/2
  // The lsb of Bouncer is (by design) 2^(E-10). Because Bouncer is
  // is very much bigger than the fp16 value, denoted by say x,
  //          2^(52+(E-10)) < Bouncer + x < 2^(53+(E-10))
  // Thus TMP := Bouncer + x  in double precision forces x to be rounded off
  // at the lsb position of 2^(E-10).
  // Consequently, the subtraction yields the desired result
  //          x_fp16_precision := TMP - Bouncer;
  // If E < -14, we are dealing with the subnormal number range, there the lsb
  // of fp16 precision is FIXED at 2^(-24) (definition of fp16).
  // Hence the Bouncer is set at 2^(52-24) = 2^(28)

  int n = 0;
  for (; n + blockSize < N; n += blockSize) {
    __m256d mA = _mm256_cvtps_pd(_mm_loadu_ps(A + n));
    __m256d mB = _mm256_cvtps_pd(_mm_loadu_ps(B + n));
    __m256d mOut = _mm256_cvtps_pd(_mm_loadu_ps(Out + n));

    mOut = _mm256_fmadd_pd(mA, mB, mOut);

    __m256i mExpv =
        _mm256_and_si256(_mm256_castpd_si256(mOut), _mm256_set1_epi64x(mask));
    mExpv = _mm256_srli_epi64(mExpv, shift_bits);
    mExpv = _mm256_sub_epi64(mExpv, _mm256_set1_epi64x(offset));

    __m256i cmp = _mm256_cmpgt_epi64(_mm256_set1_epi64x(-14), mExpv);

    __m256i mExpoBouncer = _mm256_and_si256(cmp, _mm256_set1_epi64x(28));
    mExpoBouncer = _mm256_or_si256(
        mExpoBouncer,
        _mm256_andnot_si256(
            cmp, _mm256_add_epi64(_mm256_set1_epi64x(42), mExpv)));

    __m256i mBouncer = _mm256_add_epi64(
        _mm256_set1_epi64x(dbl_threehalf),
        _mm256_slli_epi64(mExpoBouncer, shift_bits));

    mOut = _mm256_sub_pd(
        _mm256_add_pd(_mm256_castsi256_pd(mBouncer), mOut),
        _mm256_castsi256_pd(mBouncer));

    _mm_storeu_ps(Out + n, _mm256_cvtpd_ps(mOut));
  }
  // Epilogue
  for (; n < N; n++) {
    typedef union {
      uint64_t I;
      double F;
    } flint64;

    flint64 A_, B_, Out_, Bouncer;
    A_.F = A[n];
    B_.F = B[n];
    Out_.F = Out[n];

    // This is FMA in FP64
    Out_.F = std::fma(A_.F, B_.F, Out_.F);

    // We now round Out_.F to fp16 precision using a Bouncer

    // First, figure out the exponent value E of Out_.F
    int64_t expv = ((Out_.I & mask) >> shift_bits) - offset;

    // Second: create the Bouncer. To do that, we
    // first compute its exponent and then add that exponent value
    // to the exponent field of the constant 3/2.
    if (expv < -14) {
      expo_bouncer = 28;
    } else {
      expo_bouncer = 42 + expv;
    }
    Bouncer.I = dbl_threehalf + (expo_bouncer << shift_bits);

    // This is rounding to fp16 precision; add and subtract Bouncer
    Out_.F = (Bouncer.F + Out_.F) - Bouncer.F;
    Out[n] = Out_.F;
  }
}

float fmafp32_avx_emulation(float v1, float v2, float v3) {
  __m256 v1Vec = _mm256_set1_ps(v1);
  __m256 v2Vec = _mm256_set1_ps(v2);
  __m256 v3Vec = _mm256_set1_ps(v3);
  __m256 resVec = _mm256_fmadd_ps(v1Vec, v2Vec, v3Vec);
  float *result = (float *)&resVec;
  return *result;
}

} // namespace fake_fp16
