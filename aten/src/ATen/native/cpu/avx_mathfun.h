#pragma once
/*
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#include <ATen/native/cpu/Intrinsics.h>

/* The original source of this file has been modified. */
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)

#if defined(__GNUC__)
# define ALIGN32_BEG __attribute__((aligned(32)))
#elif defined(_WIN32)
# define ALIGN32_BEG __declspec(align(32))
#endif

typedef __m256  v8sf; // vector of 8 float (avx2)
typedef __m256i v8si; // vector of 8 int   (avx2)

/* declare some AVX constants -- why can't I figure a better way to do that? */
#define _PS256_CONST(Name, Val)                                            \
  static const ALIGN32_BEG float _ps256_##Name[8] = { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PI32_CONST256(Name, Val)                                            \
  static const ALIGN32_BEG int _pi32_256_##Name[8] = { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PS256_CONST_TYPE(Name, Type, Val)                                 \
  static const ALIGN32_BEG Type _ps256_##Name[8] = { Val, Val, Val, Val, Val, Val, Val, Val }

_PS256_CONST(1  , 1.0f);
_PS256_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(4, 4);
_PI32_CONST256(0x7f, 0x7f);

_PS256_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS256_CONST(cephes_log_p0, 7.0376836292E-2);
_PS256_CONST(cephes_log_p1, - 1.1514610310E-1);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1);
_PS256_CONST(cephes_log_p3, - 1.2420140846E-1);
_PS256_CONST(cephes_log_p4, + 1.4249322787E-1);
_PS256_CONST(cephes_log_p5, - 1.6668057665E-1);
_PS256_CONST(cephes_log_p6, + 2.0000714765E-1);
_PS256_CONST(cephes_log_p7, - 2.4999993993E-1);
_PS256_CONST(cephes_log_p8, + 3.3333331174E-1);
_PS256_CONST(cephes_log_q1, -2.12194440e-4);
_PS256_CONST(cephes_log_q2, 0.693359375);


/* natural logarithm computed for 8 simultaneous float
   return NaN for x <= 0
*/
inline v8sf log256_ps(v8sf x) {
  v8si imm0;
  v8sf one = *(v8sf*)_ps256_1;

  //v8sf invalid_mask = _mm256_cmple_ps(x, _mm256_setzero_ps());
  v8sf invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

  x = _mm256_max_ps(x, *(v8sf*)_ps256_min_norm_pos);  /* cut off denormalized stuff */

  // can be done with AVX2
  imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

  /* keep only the fractional part */
  x = _mm256_and_ps(x, *(v8sf*)_ps256_inv_mant_mask);
  x = _mm256_or_ps(x, *(v8sf*)_ps256_0p5);

  // this is again another AVX2 instruction
  imm0 = _mm256_sub_epi32(imm0, *(v8si*)_pi32_256_0x7f);
  v8sf e = _mm256_cvtepi32_ps(imm0);

  e = _mm256_add_ps(e, one);

  /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  //v8sf mask = _mm256_cmplt_ps(x, *(v8sf*)_ps256_cephes_SQRTHF);
  v8sf mask = _mm256_cmp_ps(x, *(v8sf*)_ps256_cephes_SQRTHF, _CMP_LT_OS);
  v8sf tmp = _mm256_and_ps(x, mask);
  x = _mm256_sub_ps(x, one);
  e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
  x = _mm256_add_ps(x, tmp);

  v8sf z = _mm256_mul_ps(x,x);

  v8sf y = *(v8sf*)_ps256_cephes_log_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p5);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p6);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p7);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p8);
  y = _mm256_mul_ps(y, x);

  y = _mm256_mul_ps(y, z);

  tmp = _mm256_mul_ps(e, *(v8sf*)_ps256_cephes_log_q1);
  y = _mm256_add_ps(y, tmp);


  tmp = _mm256_mul_ps(z, *(v8sf*)_ps256_0p5);
  y = _mm256_sub_ps(y, tmp);

  tmp = _mm256_mul_ps(e, *(v8sf*)_ps256_cephes_log_q2);
  x = _mm256_add_ps(x, y);
  x = _mm256_add_ps(x, tmp);
  x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
  return x;
}

_PS256_CONST(exp_hi,        88.3762626647949f);
_PS256_CONST(exp_lo,        -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS256_CONST(cephes_exp_C1, 0.693359375);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1);

inline v8sf exp256_ps(v8sf x) {
  v8sf tmp = _mm256_setzero_ps(), fx;
  v8si imm0;
  v8sf one = *(v8sf*)_ps256_1;

  x = _mm256_min_ps(x, *(v8sf*)_ps256_exp_hi);
  x = _mm256_max_ps(x, *(v8sf*)_ps256_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_LOG2EF);
  fx = _mm256_add_ps(fx, *(v8sf*)_ps256_0p5);

  /* how to perform a floorf with SSE: just below */
  //imm0 = _mm256_cvttps_epi32(fx);
  //tmp  = _mm256_cvtepi32_ps(imm0);

  tmp = _mm256_floor_ps(fx);

  /* if greater, subtract 1 */
  //v8sf mask = _mm256_cmpgt_ps(tmp, fx);
  v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);

  tmp = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C1);
  v8sf z = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C2);
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);

  z = _mm256_mul_ps(x,x);

  v8sf y = *(v8sf*)_ps256_cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  // another two AVX2 instructions
  imm0 = _mm256_add_epi32(imm0, *(v8si*)_pi32_256_0x7f);
  imm0 = _mm256_slli_epi32(imm0, 23);
  v8sf pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}

_PS256_CONST(minus_cephes_DP1, -0.78515625);
_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS256_CONST(sincof_p0, -1.9515295891E-4);
_PS256_CONST(sincof_p1,  8.3321608736E-3);
_PS256_CONST(sincof_p2, -1.6666654611E-1);
_PS256_CONST(coscof_p0,  2.443315711809948E-005);
_PS256_CONST(coscof_p1, -1.388731625493765E-003);
_PS256_CONST(coscof_p2,  4.166664568298827E-002);
_PS256_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI


/* evaluation of 8 sines at onces using AVX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
inline v8sf sin256_ps(v8sf x) { // any x
  v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, sign_bit, y;
  v8si imm0, imm2;

  sign_bit = x;
  /* take the absolute value */
  x = _mm256_and_ps(x, *(v8sf*)_ps256_inv_sign_mask);
  /* extract the sign bit (upper one) */
  sign_bit = _mm256_and_ps(sign_bit, *(v8sf*)_ps256_sign_mask);

  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_FOPI);

  /*
    Here we start a series of integer operations, which are in the
    realm of AVX2.
    If we don't have AVX, let's perform them using SSE2 directives
  */

  /* store the integer part of y in mm0 */
  imm2 = _mm256_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  // another two AVX2 instruction
  imm2 = _mm256_add_epi32(imm2, *(v8si*)_pi32_256_1);
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_inv1);
  y = _mm256_cvtepi32_ps(imm2);

  /* get the swap sign flag */
  imm0 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_4);
  imm0 = _mm256_slli_epi32(imm0, 29);
  /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_2);
  imm2 = _mm256_cmpeq_epi32(imm2,*(v8si*)_pi32_256_0);

  v8sf swap_sign_bit = _mm256_castsi256_ps(imm0);
  v8sf poly_mask = _mm256_castsi256_ps(imm2);
  sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(v8sf*)_ps256_minus_cephes_DP1;
  xmm2 = *(v8sf*)_ps256_minus_cephes_DP2;
  xmm3 = *(v8sf*)_ps256_minus_cephes_DP3;
  xmm1 = _mm256_mul_ps(y, xmm1);
  xmm2 = _mm256_mul_ps(y, xmm2);
  xmm3 = _mm256_mul_ps(y, xmm3);
  x = _mm256_add_ps(x, xmm1);
  x = _mm256_add_ps(x, xmm2);
  x = _mm256_add_ps(x, xmm3);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(v8sf*)_ps256_coscof_p0;
  v8sf z = _mm256_mul_ps(x,x);

  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p1);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  v8sf tmp = _mm256_mul_ps(z, *(v8sf*)_ps256_0p5);
  y = _mm256_sub_ps(y, tmp);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_1);

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  v8sf y2 = *(v8sf*)_ps256_sincof_p0;
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p1);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);

  /* select the correct result from the two polynoms */
  xmm3 = poly_mask;
  y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
  y = _mm256_andnot_ps(xmm3, y);
  y = _mm256_add_ps(y,y2);
  /* update the sign */
  y = _mm256_xor_ps(y, sign_bit);

  return y;
}

/* almost the same as sin_ps */
inline v8sf cos256_ps(v8sf x) { // any x
  v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, y;
  v8si imm0, imm2;

  /* take the absolute value */
  x = _mm256_and_ps(x, *(v8sf*)_ps256_inv_sign_mask);

  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_FOPI);

  /* store the integer part of y in mm0 */
  imm2 = _mm256_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = _mm256_add_epi32(imm2, *(v8si*)_pi32_256_1);
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_inv1);
  y = _mm256_cvtepi32_ps(imm2);
  imm2 = _mm256_sub_epi32(imm2, *(v8si*)_pi32_256_2);

  /* get the swap sign flag */
  imm0 =  _mm256_andnot_si256(imm2, *(v8si*)_pi32_256_4);
  imm0 = _mm256_slli_epi32(imm0, 29);
  /* get the polynom selection mask */
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_2);
  imm2 = _mm256_cmpeq_epi32(imm2, *(v8si*)_pi32_256_0);

  v8sf sign_bit = _mm256_castsi256_ps(imm0);
  v8sf poly_mask = _mm256_castsi256_ps(imm2);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(v8sf*)_ps256_minus_cephes_DP1;
  xmm2 = *(v8sf*)_ps256_minus_cephes_DP2;
  xmm3 = *(v8sf*)_ps256_minus_cephes_DP3;
  xmm1 = _mm256_mul_ps(y, xmm1);
  xmm2 = _mm256_mul_ps(y, xmm2);
  xmm3 = _mm256_mul_ps(y, xmm3);
  x = _mm256_add_ps(x, xmm1);
  x = _mm256_add_ps(x, xmm2);
  x = _mm256_add_ps(x, xmm3);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(v8sf*)_ps256_coscof_p0;
  v8sf z = _mm256_mul_ps(x,x);

  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p1);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  v8sf tmp = _mm256_mul_ps(z, *(v8sf*)_ps256_0p5);
  y = _mm256_sub_ps(y, tmp);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_1);

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  v8sf y2 = *(v8sf*)_ps256_sincof_p0;
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p1);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);

  /* select the correct result from the two polynoms */
  xmm3 = poly_mask;
  y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
  y = _mm256_andnot_ps(xmm3, y);
  y = _mm256_add_ps(y,y2);
  /* update the sign */
  y = _mm256_xor_ps(y, sign_bit);

  return y;
}

/* since sin256_ps and cos256_ps are almost identical, sincos256_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
inline void sincos256_ps(v8sf x, v8sf *s, v8sf *c) {

  v8sf xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
  v8si imm0, imm2, imm4;

  sign_bit_sin = x;
  /* take the absolute value */
  x = _mm256_and_ps(x, *(v8sf*)_ps256_inv_sign_mask);
  /* extract the sign bit (upper one) */
  sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(v8sf*)_ps256_sign_mask);

  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_FOPI);

  /* store the integer part of y in imm2 */
  imm2 = _mm256_cvttps_epi32(y);

  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = _mm256_add_epi32(imm2, *(v8si*)_pi32_256_1);
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_inv1);

  y = _mm256_cvtepi32_ps(imm2);
  imm4 = imm2;

  /* get the swap sign flag for the sine */
  imm0 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_4);
  imm0 = _mm256_slli_epi32(imm0, 29);
  //v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);

  /* get the polynom selection mask for the sine*/
  imm2 = _mm256_and_si256(imm2, *(v8si*)_pi32_256_2);
  imm2 = _mm256_cmpeq_epi32(imm2, *(v8si*)_pi32_256_0);
  //v8sf poly_mask = _mm256_castsi256_ps(imm2);

  v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
  v8sf poly_mask = _mm256_castsi256_ps(imm2);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(v8sf*)_ps256_minus_cephes_DP1;
  xmm2 = *(v8sf*)_ps256_minus_cephes_DP2;
  xmm3 = *(v8sf*)_ps256_minus_cephes_DP3;
  xmm1 = _mm256_mul_ps(y, xmm1);
  xmm2 = _mm256_mul_ps(y, xmm2);
  xmm3 = _mm256_mul_ps(y, xmm3);
  x = _mm256_add_ps(x, xmm1);
  x = _mm256_add_ps(x, xmm2);
  x = _mm256_add_ps(x, xmm3);

  imm4 = _mm256_sub_epi32(imm4, *(v8si*)_pi32_256_2);
  imm4 =  _mm256_andnot_si256(imm4, *(v8si*)_pi32_256_4);
  imm4 = _mm256_slli_epi32(imm4, 29);

  v8sf sign_bit_cos = _mm256_castsi256_ps(imm4);

  sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  v8sf z = _mm256_mul_ps(x,x);
  y = *(v8sf*)_ps256_coscof_p0;

  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p1);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  v8sf tmp = _mm256_mul_ps(z, *(v8sf*)_ps256_0p5);
  y = _mm256_sub_ps(y, tmp);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_1);

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  v8sf y2 = *(v8sf*)_ps256_sincof_p0;
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p1);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);

  /* select the correct result from the two polynoms */
  xmm3 = poly_mask;
  v8sf ysin2 = _mm256_and_ps(xmm3, y2);
  v8sf ysin1 = _mm256_andnot_ps(xmm3, y);
  y2 = _mm256_sub_ps(y2,ysin2);
  y = _mm256_sub_ps(y, ysin1);

  xmm1 = _mm256_add_ps(ysin1,ysin2);
  xmm2 = _mm256_add_ps(y,y2);

  /* update the sign */
  *s = _mm256_xor_ps(xmm1, sign_bit_sin);
  *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

#endif // CPU_CAPABILITY_AVX2 or CPU_CAPABILITY_AVX512

#ifdef CPU_CAPABILITY_AVX512

#if defined(__GNUC__)
# define ALIGN64_BEG __attribute__((aligned(64)))
#elif defined(_WIN32)
# define ALIGN64_BEG __declspec(align(64))
#endif

typedef __m512  v16sf; // vector of 8 float (avx2)
typedef __m512i v16si; // vector of 8 int   (avx2)

/* declare some AVX constants -- why can't I figure a better way to do that? */
#define _PS512_CONST(Name, Val)                                                                \
  static const ALIGN64_BEG float _ps512_##Name[16] = { Val, Val, Val, Val, Val, Val, Val, Val, \
                                                       Val, Val, Val, Val, Val, Val, Val, Val }
#define _PI32_CONST512(Name, Val)                                                               \
  static const ALIGN64_BEG int _pi32_512_##Name[16] = { Val, Val, Val, Val, Val, Val, Val, Val, \
                                                        Val, Val, Val, Val, Val, Val, Val, Val }
#define _PS512_CONST_TYPE(Name, Type, Val)                                                    \
  static const ALIGN64_BEG Type _ps512_##Name[16] = { Val, Val, Val, Val, Val, Val, Val, Val, \
                                                      Val, Val, Val, Val, Val, Val, Val, Val }

_PS512_CONST(1  , 1.0f);
_PS512_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS512_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS512_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS512_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS512_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS512_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST512(0, 0);
_PI32_CONST512(1, 1);
_PI32_CONST512(inv1, ~1);
_PI32_CONST512(2, 2);
_PI32_CONST512(4, 4);
_PI32_CONST512(0x7f, 0x7f);

_PS512_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS512_CONST(cephes_log_p0, 7.0376836292E-2);
_PS512_CONST(cephes_log_p1, - 1.1514610310E-1);
_PS512_CONST(cephes_log_p2, 1.1676998740E-1);
_PS512_CONST(cephes_log_p3, - 1.2420140846E-1);
_PS512_CONST(cephes_log_p4, + 1.4249322787E-1);
_PS512_CONST(cephes_log_p5, - 1.6668057665E-1);
_PS512_CONST(cephes_log_p6, + 2.0000714765E-1);
_PS512_CONST(cephes_log_p7, - 2.4999993993E-1);
_PS512_CONST(cephes_log_p8, + 3.3333331174E-1);
_PS512_CONST(cephes_log_q1, -2.12194440e-4);
_PS512_CONST(cephes_log_q2, 0.693359375);


/* natural logarithm computed for 16 simultaneous float
   return NaN for x <= 0
*/
inline v16sf log512_ps(v16sf x) {
  v16si imm0;
  v16sf one = *(v16sf*)_ps512_1;

  //v16sf invalid_mask = _mm512_cmple_ps(x, _mm512_setzero_ps());
  __mmask16 invalid_mask = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OS);
  v16si all_one = _mm512_set1_epi32(0xffffffff);
  v16si all_zero = _mm512_set1_epi32(0x00000000);
  v16sf invalid_mask_vec = _mm512_castsi512_ps(_mm512_mask_blend_epi32(invalid_mask, all_zero, all_one));

  x = _mm512_max_ps(x, *(v16sf*)_ps512_min_norm_pos);  /* cut off denormalized stuff */

  // can be done with AVX512
  imm0 = _mm512_srli_epi32(_mm512_castps_si512(x), 23);

  /* keep only the fractional part */
  x = _mm512_and_ps(x, *(v16sf*)_ps512_inv_mant_mask);
  x = _mm512_or_ps(x, *(v16sf*)_ps512_0p5);

  // this is again another AVX512 instruction
  imm0 = _mm512_sub_epi32(imm0, *(v16si*)_pi32_512_0x7f);
  v16sf e = _mm512_cvtepi32_ps(imm0);

  e = _mm512_add_ps(e, one);

  /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  //v16sf mask = _mm512_cmplt_ps(x, *(v16sf*)_ps512_cephes_SQRTHF);
  __mmask16 mask = _mm512_cmp_ps_mask(x, *(v16sf*)_ps512_cephes_SQRTHF, _CMP_LT_OS);
  v16sf mask_vector = _mm512_castsi512_ps(_mm512_mask_blend_epi32(mask, all_zero, all_one));
  v16sf tmp = _mm512_and_ps(x, mask_vector);
  x = _mm512_sub_ps(x, one);
  e = _mm512_sub_ps(e, _mm512_and_ps(one, mask_vector));
  x = _mm512_add_ps(x, tmp);

  v16sf z = _mm512_mul_ps(x,x);

  v16sf y = *(v16sf*)_ps512_cephes_log_p0;
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_log_p1);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_log_p2);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_log_p3);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_log_p4);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_log_p5);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_log_p6);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_log_p7);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_log_p8);
  y = _mm512_mul_ps(y, x);

  y = _mm512_mul_ps(y, z);

  tmp = _mm512_mul_ps(e, *(v16sf*)_ps512_cephes_log_q1);
  y = _mm512_add_ps(y, tmp);


  tmp = _mm512_mul_ps(z, *(v16sf*)_ps512_0p5);
  y = _mm512_sub_ps(y, tmp);

  tmp = _mm512_mul_ps(e, *(v16sf*)_ps512_cephes_log_q2);
  x = _mm512_add_ps(x, y);
  x = _mm512_add_ps(x, tmp);
  x = _mm512_or_ps(x, invalid_mask_vec); // negative arg will be NAN
  return x;
}

_PS512_CONST(exp_hi,        88.3762626647949f);
_PS512_CONST(exp_lo,        -88.3762626647949f);

_PS512_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS512_CONST(cephes_exp_C1, 0.693359375);
_PS512_CONST(cephes_exp_C2, -2.12194440e-4);

_PS512_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS512_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS512_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS512_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS512_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS512_CONST(cephes_exp_p5, 5.0000001201E-1);

inline v16sf exp512_ps(v16sf x) {
  v16sf tmp = _mm512_setzero_ps(), fx;
  v16si imm0;
  v16sf one = *(v16sf*)_ps512_1;

  x = _mm512_min_ps(x, *(v16sf*)_ps512_exp_hi);
  x = _mm512_max_ps(x, *(v16sf*)_ps512_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm512_mul_ps(x, *(v16sf*)_ps512_cephes_LOG2EF);
  fx = _mm512_add_ps(fx, *(v16sf*)_ps512_0p5);

  /* how to perform a floorf with SSE: just below */
  //imm0 = _mm512_cvttps_epi32(fx);
  //tmp  = _mm512_cvtepi32_ps(imm0);

  tmp = _mm512_floor_ps(fx);

  /* if greater, subtract 1 */
  //v16sf mask = _mm512_cmpgt_ps(tmp, fx);
  __mmask16 mask = _mm512_cmp_ps_mask(tmp, fx, _CMP_GT_OS);
  v16si all_one = _mm512_set1_epi32(0xffffffff);
  v16si all_zero = _mm512_set1_epi32(0x00000000);
  v16sf mask_vector = _mm512_castsi512_ps(_mm512_mask_blend_epi32(mask, all_zero, all_one));
  mask_vector = _mm512_and_ps(mask_vector, one);
  fx = _mm512_sub_ps(tmp, mask_vector);

  tmp = _mm512_mul_ps(fx, *(v16sf*)_ps512_cephes_exp_C1);
  v16sf z = _mm512_mul_ps(fx, *(v16sf*)_ps512_cephes_exp_C2);
  x = _mm512_sub_ps(x, tmp);
  x = _mm512_sub_ps(x, z);

  z = _mm512_mul_ps(x,x);

  v16sf y = *(v16sf*)_ps512_cephes_exp_p0;
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_exp_p1);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_exp_p2);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_exp_p3);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_exp_p4);
  y = _mm512_mul_ps(y, x);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_cephes_exp_p5);
  y = _mm512_mul_ps(y, z);
  y = _mm512_add_ps(y, x);
  y = _mm512_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm512_cvttps_epi32(fx);
  // another two AVX512 instructions
  imm0 = _mm512_add_epi32(imm0, *(v16si*)_pi32_512_0x7f);
  imm0 = _mm512_slli_epi32(imm0, 23);
  v16sf pow2n = _mm512_castsi512_ps(imm0);
  y = _mm512_mul_ps(y, pow2n);
  return y;
}

_PS512_CONST(minus_cephes_DP1, -0.78515625);
_PS512_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS512_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS512_CONST(sincof_p0, -1.9515295891E-4);
_PS512_CONST(sincof_p1,  8.3321608736E-3);
_PS512_CONST(sincof_p2, -1.6666654611E-1);
_PS512_CONST(coscof_p0,  2.443315711809948E-005);
_PS512_CONST(coscof_p1, -1.388731625493765E-003);
_PS512_CONST(coscof_p2,  4.166664568298827E-002);
_PS512_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI


/* evaluation of 16 sines at onces using AVX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
inline v16sf sin512_ps(v16sf x) { // any x
  v16sf zmm1, zmm2 = _mm512_setzero_ps(), zmm3, sign_bit, y;
  v16si imm0, imm2;

  sign_bit = x;
  /* take the absolute value */
  x = _mm512_and_ps(x, *(v16sf*)_ps512_inv_sign_mask);
  /* extract the sign bit (upper one) */
  sign_bit = _mm512_and_ps(sign_bit, *(v16sf*)_ps512_sign_mask);

  /* scale by 4/Pi */
  y = _mm512_mul_ps(x, *(v16sf*)_ps512_cephes_FOPI);

  /*
    Here we start a series of integer operations, which are in the
    realm of AVX512.
    If we don't have AVX, let's perform them using SSE2 directives
  */

  /* store the integer part of y in mm0 */
  imm2 = _mm512_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  // another two AVX512 instruction
  imm2 = _mm512_add_epi32(imm2, *(v16si*)_pi32_512_1);
  imm2 = _mm512_and_si512(imm2, *(v16si*)_pi32_512_inv1);
  y = _mm512_cvtepi32_ps(imm2);

  /* get the swap sign flag */
  imm0 = _mm512_and_si512(imm2, *(v16si*)_pi32_512_4);
  imm0 = _mm512_slli_epi32(imm0, 29);
  /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
  imm2 = _mm512_and_si512(imm2, *(v16si*)_pi32_512_2);
  __mmask16 imm2_mask = _mm512_cmpeq_epi32_mask(imm2,*(v16si*)_pi32_512_0);
  v16si all_one = _mm512_set1_epi32(0xffffffff);
  v16si all_zero = _mm512_set1_epi32(0x00000000);
  imm2 = _mm512_mask_blend_epi32(imm2_mask, all_zero, all_one);

  v16sf swap_sign_bit = _mm512_castsi512_ps(imm0);
  v16sf poly_mask = _mm512_castsi512_ps(imm2);
  sign_bit = _mm512_xor_ps(sign_bit, swap_sign_bit);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  zmm1 = *(v16sf*)_ps512_minus_cephes_DP1;
  zmm2 = *(v16sf*)_ps512_minus_cephes_DP2;
  zmm3 = *(v16sf*)_ps512_minus_cephes_DP3;
  zmm1 = _mm512_mul_ps(y, zmm1);
  zmm2 = _mm512_mul_ps(y, zmm2);
  zmm3 = _mm512_mul_ps(y, zmm3);
  x = _mm512_add_ps(x, zmm1);
  x = _mm512_add_ps(x, zmm2);
  x = _mm512_add_ps(x, zmm3);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(v16sf*)_ps512_coscof_p0;
  v16sf z = _mm512_mul_ps(x,x);

  y = _mm512_mul_ps(y, z);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_coscof_p1);
  y = _mm512_mul_ps(y, z);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_coscof_p2);
  y = _mm512_mul_ps(y, z);
  y = _mm512_mul_ps(y, z);
  v16sf tmp = _mm512_mul_ps(z, *(v16sf*)_ps512_0p5);
  y = _mm512_sub_ps(y, tmp);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_1);

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  v16sf y2 = *(v16sf*)_ps512_sincof_p0;
  y2 = _mm512_mul_ps(y2, z);
  y2 = _mm512_add_ps(y2, *(v16sf*)_ps512_sincof_p1);
  y2 = _mm512_mul_ps(y2, z);
  y2 = _mm512_add_ps(y2, *(v16sf*)_ps512_sincof_p2);
  y2 = _mm512_mul_ps(y2, z);
  y2 = _mm512_mul_ps(y2, x);
  y2 = _mm512_add_ps(y2, x);

  /* select the correct result from the two polynoms */
  zmm3 = poly_mask;
  y2 = _mm512_and_ps(zmm3, y2); //, zmm3);
  y = _mm512_andnot_ps(zmm3, y);
  y = _mm512_add_ps(y,y2);
  /* update the sign */
  y = _mm512_xor_ps(y, sign_bit);

  return y;
}

/* almost the same as sin_ps */
inline v16sf cos512_ps(v16sf x) { // any x
  v16sf zmm1, zmm2 = _mm512_setzero_ps(), zmm3, y;
  v16si imm0, imm2;

  /* take the absolute value */
  x = _mm512_and_ps(x, *(v16sf*)_ps512_inv_sign_mask);

  /* scale by 4/Pi */
  y = _mm512_mul_ps(x, *(v16sf*)_ps512_cephes_FOPI);

  /* store the integer part of y in mm0 */
  imm2 = _mm512_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = _mm512_add_epi32(imm2, *(v16si*)_pi32_512_1);
  imm2 = _mm512_and_si512(imm2, *(v16si*)_pi32_512_inv1);
  y = _mm512_cvtepi32_ps(imm2);
  imm2 = _mm512_sub_epi32(imm2, *(v16si*)_pi32_512_2);

  /* get the swap sign flag */
  imm0 =  _mm512_andnot_si512(imm2, *(v16si*)_pi32_512_4);
  imm0 = _mm512_slli_epi32(imm0, 29);
  /* get the polynom selection mask */
  imm2 = _mm512_and_si512(imm2, *(v16si*)_pi32_512_2);
  __mmask16 imm2_mask = _mm512_cmpeq_epi32_mask(imm2, *(v16si*)_pi32_512_0);
  const v16si all_ones = _mm512_set1_epi32(0xffffffff);
  const v16si all_zero = _mm512_set1_epi32(0x00000000);
  imm2 = _mm512_mask_blend_epi32(imm2_mask, all_zero, all_ones);

  v16sf sign_bit = _mm512_castsi512_ps(imm0);
  v16sf poly_mask = _mm512_castsi512_ps(imm2);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  zmm1 = *(v16sf*)_ps512_minus_cephes_DP1;
  zmm2 = *(v16sf*)_ps512_minus_cephes_DP2;
  zmm3 = *(v16sf*)_ps512_minus_cephes_DP3;
  zmm1 = _mm512_mul_ps(y, zmm1);
  zmm2 = _mm512_mul_ps(y, zmm2);
  zmm3 = _mm512_mul_ps(y, zmm3);
  x = _mm512_add_ps(x, zmm1);
  x = _mm512_add_ps(x, zmm2);
  x = _mm512_add_ps(x, zmm3);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(v16sf*)_ps512_coscof_p0;
  v16sf z = _mm512_mul_ps(x,x);

  y = _mm512_mul_ps(y, z);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_coscof_p1);
  y = _mm512_mul_ps(y, z);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_coscof_p2);
  y = _mm512_mul_ps(y, z);
  y = _mm512_mul_ps(y, z);
  v16sf tmp = _mm512_mul_ps(z, *(v16sf*)_ps512_0p5);
  y = _mm512_sub_ps(y, tmp);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_1);

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  v16sf y2 = *(v16sf*)_ps512_sincof_p0;
  y2 = _mm512_mul_ps(y2, z);
  y2 = _mm512_add_ps(y2, *(v16sf*)_ps512_sincof_p1);
  y2 = _mm512_mul_ps(y2, z);
  y2 = _mm512_add_ps(y2, *(v16sf*)_ps512_sincof_p2);
  y2 = _mm512_mul_ps(y2, z);
  y2 = _mm512_mul_ps(y2, x);
  y2 = _mm512_add_ps(y2, x);

  /* select the correct result from the two polynoms */
  zmm3 = poly_mask;
  y2 = _mm512_and_ps(zmm3, y2); //, zmm3);
  y = _mm512_andnot_ps(zmm3, y);
  y = _mm512_add_ps(y,y2);
  /* update the sign */
  y = _mm512_xor_ps(y, sign_bit);

  return y;
}

/* since sin512_ps and cos512_ps are almost identical, sincos512_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
inline void sincos512_ps(v16sf x, v16sf *s, v16sf *c) {

  v16sf zmm1, zmm2, zmm3 = _mm512_setzero_ps(), sign_bit_sin, y;
  v16si imm0, imm2, imm4;

  sign_bit_sin = x;
  /* take the absolute value */
  x = _mm512_and_ps(x, *(v16sf*)_ps512_inv_sign_mask);
  /* extract the sign bit (upper one) */
  sign_bit_sin = _mm512_and_ps(sign_bit_sin, *(v16sf*)_ps512_sign_mask);

  /* scale by 4/Pi */
  y = _mm512_mul_ps(x, *(v16sf*)_ps512_cephes_FOPI);

  /* store the integer part of y in imm2 */
  imm2 = _mm512_cvttps_epi32(y);

  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = _mm512_add_epi32(imm2, *(v16si*)_pi32_512_1);
  imm2 = _mm512_and_si512(imm2, *(v16si*)_pi32_512_inv1);

  y = _mm512_cvtepi32_ps(imm2);
  imm4 = imm2;

  /* get the swap sign flag for the sine */
  imm0 = _mm512_and_si512(imm2, *(v16si*)_pi32_512_4);
  imm0 = _mm512_slli_epi32(imm0, 29);
  //v16sf swap_sign_bit_sin = _mm512_castsi512_ps(imm0);

  /* get the polynom selection mask for the sine*/
  imm2 = _mm512_and_si512(imm2, *(v16si*)_pi32_512_2);
  v16si all_one = _mm512_set1_epi32(0xffffffff);
  v16si all_zero = _mm512_set1_epi32(0x00000000);
  __mmask16 imm2_mask = _mm512_cmpeq_epi32_mask(imm2, *(v16si*)_pi32_512_0);
  imm2 = _mm512_mask_blend_epi32(imm2_mask, all_zero, all_one);
  //v16sf poly_mask = _mm512_castsi512_ps(imm2);

  v16sf swap_sign_bit_sin = _mm512_castsi512_ps(imm0);
  v16sf poly_mask = _mm512_castsi512_ps(imm2);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  zmm1 = *(v16sf*)_ps512_minus_cephes_DP1;
  zmm2 = *(v16sf*)_ps512_minus_cephes_DP2;
  zmm3 = *(v16sf*)_ps512_minus_cephes_DP3;
  zmm1 = _mm512_mul_ps(y, zmm1);
  zmm2 = _mm512_mul_ps(y, zmm2);
  zmm3 = _mm512_mul_ps(y, zmm3);
  x = _mm512_add_ps(x, zmm1);
  x = _mm512_add_ps(x, zmm2);
  x = _mm512_add_ps(x, zmm3);

  imm4 = _mm512_sub_epi32(imm4, *(v16si*)_pi32_512_2);
  imm4 =  _mm512_andnot_si512(imm4, *(v16si*)_pi32_512_4);
  imm4 = _mm512_slli_epi32(imm4, 29);

  v16sf sign_bit_cos = _mm512_castsi512_ps(imm4);

  sign_bit_sin = _mm512_xor_ps(sign_bit_sin, swap_sign_bit_sin);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  v16sf z = _mm512_mul_ps(x,x);
  y = *(v16sf*)_ps512_coscof_p0;

  y = _mm512_mul_ps(y, z);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_coscof_p1);
  y = _mm512_mul_ps(y, z);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_coscof_p2);
  y = _mm512_mul_ps(y, z);
  y = _mm512_mul_ps(y, z);
  v16sf tmp = _mm512_mul_ps(z, *(v16sf*)_ps512_0p5);
  y = _mm512_sub_ps(y, tmp);
  y = _mm512_add_ps(y, *(v16sf*)_ps512_1);

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  v16sf y2 = *(v16sf*)_ps512_sincof_p0;
  y2 = _mm512_mul_ps(y2, z);
  y2 = _mm512_add_ps(y2, *(v16sf*)_ps512_sincof_p1);
  y2 = _mm512_mul_ps(y2, z);
  y2 = _mm512_add_ps(y2, *(v16sf*)_ps512_sincof_p2);
  y2 = _mm512_mul_ps(y2, z);
  y2 = _mm512_mul_ps(y2, x);
  y2 = _mm512_add_ps(y2, x);

  /* select the correct result from the two polynoms */
  zmm3 = poly_mask;
  v16sf ysin2 = _mm512_and_ps(zmm3, y2);
  v16sf ysin1 = _mm512_andnot_ps(zmm3, y);
  y2 = _mm512_sub_ps(y2,ysin2);
  y = _mm512_sub_ps(y, ysin1);

  zmm1 = _mm512_add_ps(ysin1,ysin2);
  zmm2 = _mm512_add_ps(y,y2);

  /* update the sign */
  *s = _mm512_xor_ps(zmm1, sign_bit_sin);
  *c = _mm512_xor_ps(zmm2, sign_bit_cos);
}

#endif // CPU_CAPABILITY_AVX512
