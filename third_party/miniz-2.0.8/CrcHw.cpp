/*
 * Copyright 2016 Eric Biggers
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/*
 * CRC-32 folding with PCLMULQDQ.
 *
 * The basic idea is to repeatedly "fold" each 512 bits into the next
 * 512 bits, producing an abbreviated message which is congruent the
 * original message modulo the generator polynomial G(x).
 *
 * Folding each 512 bits is implemented as eight 64-bit folds, each of
 * which uses one carryless multiplication instruction.  It's expected
 * that CPUs may be able to execute some of these multiplications in
 * parallel.
 *
 * Explanation of "folding": let A(x) be 64 bits from the message, and
 * let B(x) be 95 bits from a constant distance D later in the
 * message.  The relevant portion of the message can be written as:
 *
 *      M(x) = A(x)*x^D + B(x)
 *
 * ... where + and * represent addition and multiplication,
 * respectively, of polynomials over GF(2).  Note that when
 * implemented on a computer, these operations are equivalent to XOR
 * and carryless multiplication, respectively.
 *
 * For the purpose of CRC calculation, only the remainder modulo the
 * generator polynomial G(x) matters:
 *
 * M(x) mod G(x) = (A(x)*x^D + B(x)) mod G(x)
 *
 * Since the modulo operation can be applied anywhere in a sequence of
 * additions and multiplications without affecting the result, this is
 * equivalent to:
 *
 * M(x) mod G(x) = (A(x)*(x^D mod G(x)) + B(x)) mod G(x)
 *
 * For any D, 'x^D mod G(x)' will be a polynomial with maximum degree
 * 31, i.e.  a 32-bit quantity.  So 'A(x) * (x^D mod G(x))' is
 * equivalent to a carryless multiplication of a 64-bit quantity by a
 * 32-bit quantity, producing a 95-bit product.  Then, adding
 * (XOR-ing) the product to B(x) produces a polynomial with the same
 * length as B(x) but with the same remainder as 'A(x)*x^D + B(x)'.
 * This is the basic fold operation with 64 bits.
 *
 * Note that the carryless multiplication instruction PCLMULQDQ
 * actually takes two 64-bit inputs and produces a 127-bit product in
 * the low-order bits of a 128-bit XMM register.  This works fine, but
 * care must be taken to account for "bit endianness".  With the CRC
 * version implemented here, bits are always ordered such that the
 * lowest-order bit represents the coefficient of highest power of x
 * and the highest-order bit represents the coefficient of the lowest
 * power of x.  This is backwards from the more intuitive order.
 * Still, carryless multiplication works essentially the same either
 * way.  It just must be accounted for that when we XOR the 95-bit
 * product in the low-order 95 bits of a 128-bit XMM register into
 * 128-bits of later data held in another XMM register, we'll really
 * be XOR-ing the product into the mathematically higher degree end of
 * those later bits, not the lower degree end as may be expected.
 *
 * So given that caveat and the fact that we process 512 bits per
 * iteration, the 'D' values we need for the two 64-bit halves of each
 * 128 bits of data are:
 *
 * D = (512 + 95) - 64 for the higher-degree half of each 128
 *                 bits, i.e. the lower order bits in
 *                 the XMM register
 *
 *    D = (512 + 95) - 128 for the lower-degree half of each 128
 *                 bits, i.e. the higher order bits in
 *                 the XMM register
 *
 * The required 'x^D mod G(x)' values were precomputed.
 *
 * When <= 512 bits remain in the message, we finish up by folding
 * across smaller distances.  This works similarly; the distance D is
 * just different, so different constant multipliers must be used.
 * Finally, once the remaining message is just 64 bits, it is is
 * reduced to the CRC-32 using Barrett reduction (explained later).
 *
 * For more information see the original paper from Intel: "Fast CRC
 *    Computation for Generic Polynomials Using PCLMULQDQ
 *    Instruction" December 2009
 *    http://www.intel.com/content/dam/www/public/us/en/documents/
 *    white-papers/
 *    fast-crc-computation-generic-polynomials-pclmulqdq-paper.pdf
 */

#if defined(__SSE4_2__)
#include <emmintrin.h>
#include <nmmintrin.h>
#include <x86intrin.h>
#endif
#include "miniz.h"
#include <algorithm>

namespace {
uint32_t crc32_sw(const uint8_t* data, size_t nbytes, uint32_t startingChecksum);

#if defined(__SSE4_2__)
uint32_t
crc32_hw_aligned(uint32_t remainder, const __m128i* p, size_t vec_count) {
  /* Constants precomputed by gen_crc32_multipliers.c.  Do not edit! */
  const __m128i multipliers_4 = _mm_set_epi32(0, 0x1D9513D7, 0, 0x8F352D95);
  const __m128i multipliers_2 = _mm_set_epi32(0, 0x81256527, 0, 0xF1DA05AA);
  const __m128i multipliers_1 = _mm_set_epi32(0, 0xCCAA009E, 0, 0xAE689191);
  const __m128i final_multiplier = _mm_set_epi32(0, 0, 0, 0xB8BC6765);
  const __m128i mask32 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
  const __m128i barrett_reduction_constants =
      _mm_set_epi32(0x1, 0xDB710641, 0x1, 0xF7011641);

  const __m128i* const end = p + vec_count;
  const __m128i* const end512 = p + (vec_count & ~3);
  __m128i x0, x1, x2, x3;

  /*
   * Account for the current 'remainder', i.e. the CRC of the part of
   * the message already processed.  Explanation: rewrite the message
   * polynomial M(x) in terms of the first part A(x), the second part
   * B(x), and the length of the second part in bits |B(x)| >= 32:
   *
   *    M(x) = A(x)*x^|B(x)| + B(x)
   *
   * Then the CRC of M(x) is:
   *
   *    CRC(M(x)) = CRC(A(x)*x^|B(x)| + B(x))
   *              = CRC(A(x)*x^32*x^(|B(x)| - 32) + B(x))
   *              = CRC(CRC(A(x))*x^(|B(x)| - 32) + B(x))
   *
   * Note: all arithmetic is modulo G(x), the generator polynomial; that's
   * why A(x)*x^32 can be replaced with CRC(A(x)) = A(x)*x^32 mod G(x).
   *
   * So the CRC of the full message is the CRC of the second part of the
   * message where the first 32 bits of the second part of the message
   * have been XOR'ed with the CRC of the first part of the message.
   */
  x0 = *p++;
  x0 = _mm_xor_si128(x0, _mm_set_epi32(0, 0, 0, remainder));

  if (p > end512) /* only 128, 256, or 384 bits of input? */
    goto _128_bits_at_a_time;
  x1 = *p++;
  x2 = *p++;
  x3 = *p++;

  /* Fold 512 bits at a time */
  for (; p != end512; p += 4) {
    __m128i y0, y1, y2, y3;

    y0 = p[0];
    y1 = p[1];
    y2 = p[2];
    y3 = p[3];

    /*
     * Note: the immediate constant for PCLMULQDQ specifies which
     * 64-bit halves of the 128-bit vectors to multiply:
     *
     * 0x00 means low halves (higher degree polynomial terms for us)
     * 0x11 means high halves (lower degree polynomial terms for us)
     */
    y0 = _mm_xor_si128(y0, _mm_clmulepi64_si128(x0, multipliers_4, 0x00));
    y1 = _mm_xor_si128(y1, _mm_clmulepi64_si128(x1, multipliers_4, 0x00));
    y2 = _mm_xor_si128(y2, _mm_clmulepi64_si128(x2, multipliers_4, 0x00));
    y3 = _mm_xor_si128(y3, _mm_clmulepi64_si128(x3, multipliers_4, 0x00));
    y0 = _mm_xor_si128(y0, _mm_clmulepi64_si128(x0, multipliers_4, 0x11));
    y1 = _mm_xor_si128(y1, _mm_clmulepi64_si128(x1, multipliers_4, 0x11));
    y2 = _mm_xor_si128(y2, _mm_clmulepi64_si128(x2, multipliers_4, 0x11));
    y3 = _mm_xor_si128(y3, _mm_clmulepi64_si128(x3, multipliers_4, 0x11));

    x0 = y0;
    x1 = y1;
    x2 = y2;
    x3 = y3;
  }

  /* Fold 512 bits => 128 bits */
  x2 = _mm_xor_si128(x2, _mm_clmulepi64_si128(x0, multipliers_2, 0x00));
  x3 = _mm_xor_si128(x3, _mm_clmulepi64_si128(x1, multipliers_2, 0x00));
  x2 = _mm_xor_si128(x2, _mm_clmulepi64_si128(x0, multipliers_2, 0x11));
  x3 = _mm_xor_si128(x3, _mm_clmulepi64_si128(x1, multipliers_2, 0x11));
  x3 = _mm_xor_si128(x3, _mm_clmulepi64_si128(x2, multipliers_1, 0x00));
  x3 = _mm_xor_si128(x3, _mm_clmulepi64_si128(x2, multipliers_1, 0x11));
  x0 = x3;

_128_bits_at_a_time:
  while (p != end) {
    /* Fold 128 bits into next 128 bits */
    x1 = *p++;
    x1 = _mm_xor_si128(x1, _mm_clmulepi64_si128(x0, multipliers_1, 0x00));
    x1 = _mm_xor_si128(x1, _mm_clmulepi64_si128(x0, multipliers_1, 0x11));
    x0 = x1;
  }

  /* Now there are just 128 bits left, stored in 'x0'. */

  /*
   * Fold 128 => 96 bits.  This also implicitly appends 32 zero bits,
   * which is equivalent to multiplying by x^32.  This is needed because
   * the CRC is defined as M(x)*x^32 mod G(x), not just M(x) mod G(x).
   */
  x0 = _mm_xor_si128(
      _mm_srli_si128(x0, 8), _mm_clmulepi64_si128(x0, multipliers_1, 0x10));

  /* Fold 96 => 64 bits */
  x0 = _mm_xor_si128(
      _mm_srli_si128(x0, 4),
      _mm_clmulepi64_si128(_mm_and_si128(x0, mask32), final_multiplier, 0x00));

  /*
   * Finally, reduce 64 => 32 bits using Barrett reduction.
   *
   * Let M(x) = A(x)*x^32 + B(x) be the remaining message.  The goal is to
   * compute R(x) = M(x) mod G(x).  Since degree(B(x)) < degree(G(x)):
   *
   *    R(x) = (A(x)*x^32 + B(x)) mod G(x)
   *         = (A(x)*x^32) mod G(x) + B(x)
   *
   * Then, by the Division Algorithm there exists a unique q(x) such that:
   *
   *    A(x)*x^32 mod G(x) = A(x)*x^32 - q(x)*G(x)
   *
   * Since the left-hand side is of maximum degree 31, the right-hand side
   * must be too.  This implies that we can apply 'mod x^32' to the
   * right-hand side without changing its value:
   *
   *    (A(x)*x^32 - q(x)*G(x)) mod x^32 = q(x)*G(x) mod x^32
   *
   * Note that '+' is equivalent to '-' in polynomials over GF(2).
   *
   * We also know that:
   *
   *                  / A(x)*x^32 \
   *    q(x) = floor (  ---------  )
   *                  \    G(x)   /
   *
   * To compute this efficiently, we can multiply the top and bottom by
   * x^32 and move the division by G(x) to the top:
   *
   *                  / A(x) * floor(x^64 / G(x)) \
   *    q(x) = floor (  -------------------------  )
   *                  \           x^32            /
   *
   * Note that floor(x^64 / G(x)) is a constant.
   *
   * So finally we have:
   *
   *                              / A(x) * floor(x^64 / G(x)) \
   *    R(x) = B(x) + G(x)*floor (  -------------------------  )
   *                              \           x^32            /
   */
  x1 = x0;
  x0 = _mm_clmulepi64_si128(
      _mm_and_si128(x0, mask32), barrett_reduction_constants, 0x00);
  x0 = _mm_clmulepi64_si128(
      _mm_and_si128(x0, mask32), barrett_reduction_constants, 0x10);
  return _mm_cvtsi128_si32(_mm_srli_si128(_mm_xor_si128(x0, x1), 4));
}

// Fast SIMD implementation of CRC-32 for x86 with pclmul
uint32_t
crc32_hw(const uint8_t* data, size_t nbytes, uint32_t startingChecksum) {
  uint32_t sum = startingChecksum;
  size_t offset = 0;

  // Process unaligned bytes
  if ((uintptr_t)data & 15) {
    size_t limit = std::min(nbytes, -(uintptr_t)data & 15);
    sum = crc32_sw(data, limit, sum);
    offset += limit;
    nbytes -= limit;
  }

  if (nbytes >= 16) {
    sum = crc32_hw_aligned(sum, (const __m128i*)(data + offset), nbytes / 16);
    offset += nbytes & ~15;
    nbytes &= 15;
  }

  // Remaining unaligned bytes
  return crc32_sw(data + offset, nbytes, sum);
}

bool crc32_hw_supported() {
  static bool kSupported = __builtin_cpu_supports("sse4.2");
  return kSupported;
}
#else
bool crc32_hw_supported() {
  return false;
}

uint32_t
crc32_hw(const uint8_t* data, size_t nbytes, uint32_t startingChecksum) {
  return 0;
}
#endif

/**************************************************************************
 *
 * Copyright 2013-2014 RAD Game Tools and Valve Software
 * Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 **************************************************************************/
uint32_t crc32_sw(const uint8_t *ptr, size_t buf_len, uint32_t crc)
{
    static const uint32_t s_crc_table[256] =
        {
          0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535,
          0x9E6495A3, 0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD,
          0xE7B82D07, 0x90BF1D91, 0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D,
          0x6DDDE4EB, 0xF4D4B551, 0x83D385C7, 0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC,
          0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4,
          0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C,
          0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59, 0x26D930AC,
          0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
          0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB,
          0xB6662D3D, 0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F,
          0x9FBFE4A5, 0xE8B8D433, 0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB,
          0x086D3D2D, 0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
          0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA,
          0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65, 0x4DB26158, 0x3AB551CE,
          0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A,
          0x346ED9FC, 0xAD678846, 0xDA60B8D0, 0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9,
          0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409,
          0xCE61E49F, 0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81,
          0xB7BD5C3B, 0xC0BA6CAD, 0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A, 0xEAD54739,
          0x9DD277AF, 0x04DB2615, 0x73DC1683, 0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8,
          0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1, 0xF00F9344, 0x8708A3D2, 0x1E01F268,
          0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7, 0xFED41B76, 0x89D32BE0,
          0x10DA7A5A, 0x67DD4ACC, 0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5, 0xD6D6A3E8,
          0xA1D1937E, 0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
          0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF,
          0x4669BE79, 0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236, 0xCC0C7795, 0xBB0B4703,
          0x220216B9, 0x5505262F, 0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7,
          0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D, 0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A,
          0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713, 0x95BF4A82, 0xE2B87A14, 0x7BB12BAE,
          0x0CB61B38, 0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21, 0x86D3D2D4, 0xF1D4E242,
          0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777, 0x88085AE6,
          0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45,
          0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2, 0xA7672661, 0xD06016F7, 0x4969474D,
          0x3E6E77DB, 0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5,
          0x47B2CF7F, 0x30B5FFE9, 0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605,
          0xCDD70693, 0x54DE5729, 0x23D967BF, 0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
          0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D
        };

    uint32_t crc32 = (uint32_t)~crc ^ 0xFFFFFFFF;
    const uint8_t *pByte_buf = (const uint8_t *)ptr;

    while (buf_len >= 4)
    {
        crc32 = (crc32 >> 8) ^ s_crc_table[(crc32 ^ pByte_buf[0]) & 0xFF];
        crc32 = (crc32 >> 8) ^ s_crc_table[(crc32 ^ pByte_buf[1]) & 0xFF];
        crc32 = (crc32 >> 8) ^ s_crc_table[(crc32 ^ pByte_buf[2]) & 0xFF];
        crc32 = (crc32 >> 8) ^ s_crc_table[(crc32 ^ pByte_buf[3]) & 0xFF];
        pByte_buf += 4;
        buf_len -= 4;
    }

    while (buf_len)
    {
        crc32 = (crc32 >> 8) ^ s_crc_table[(crc32 ^ pByte_buf[0]) & 0xFF];
        ++pByte_buf;
        --buf_len;
    }

    return crc32;
}

uint32_t crc32(const uint8_t* data, size_t nbytes, uint32_t startingChecksum) {
  if (crc32_hw_supported()) {
    return crc32_hw(data, nbytes, startingChecksum);
  } else {
    return crc32_sw(data, nbytes, startingChecksum);
  }
}
};

// API exported to miniz
extern "C" {
mz_ulong mz_crc32(mz_ulong crc, const mz_uint8* ptr, size_t buf_len) {
  // The ~ operators are needed to translate between crc conventions
  // used in miniz vs. the implementation above.
  //
  // It  appears that the miniz.c code uses 0 as its initial crc,
  // and ~'s the result, while the code above assumes ~0 as its initial
  // crc, and doesn't ~ the result. Hence, the ~ are necessary to make
  // crc32() results match up. Note that in chained calls, the paired ~
  // operators cancel out.
  return ~crc32(ptr, buf_len, ~crc);
}
}
