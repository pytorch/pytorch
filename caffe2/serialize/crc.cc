#include "miniz.h"
#include <iostream>

#include "caffe2/serialize/boost_crc.h"
#include "caffe2/serialize/folly_cpu_id.h"
#include <algorithm>
#include <stdexcept>
#include <array>
#include <c10/util/llvmMathExtras.h>

#if defined(__SSE4_2__)
#include <emmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#endif


namespace detail {

// Standard galois-field multiply.  The only modification is that a,
// b, m, and p are all bit-reflected.
//
// https://en.wikipedia.org/wiki/Finite_field_arithmetic
static constexpr uint32_t
gf_multiply_sw_1(size_t i, uint32_t p, uint32_t a, uint32_t b, uint32_t m) {
  // clang-format off
  return i == 32 ? p : gf_multiply_sw_1(
      /* i = */ i + 1,
      /* p = */ p ^ (-((b >> 31) & 1) & a),
      /* a = */ (a >> 1) ^ (-(a & 1) & m),
      /* b = */ b << 1,
      /* m = */ m);
  // clang-format on
}
static constexpr uint32_t gf_multiply_sw(uint32_t a, uint32_t b, uint32_t m) {
  return gf_multiply_sw_1(/* i = */ 0, /* p = */ 0, a, b, m);
}

static constexpr uint32_t gf_square_sw(uint32_t a, uint32_t m) {
  return gf_multiply_sw(a, a, m);
}

namespace {

template <size_t i, uint32_t m>
struct gf_powers_memo {
  static constexpr uint32_t value =
      gf_square_sw(gf_powers_memo<i - 1, m>::value, m);
};
template <uint32_t m>
struct gf_powers_memo<0, m> {
  static constexpr uint32_t value = m;
};

template <std::size_t... Is>
using index_sequence = std::integer_sequence<std::size_t, Is...>;


template <uint32_t m>
struct gf_powers_make {
  template <size_t... i>
  constexpr auto operator()(std::index_sequence<i...>) const {
    return std::array<uint32_t, sizeof...(i)>{{gf_powers_memo<i, m>::value...}};
  }
};

} 

static uint32_t gf_multiply_crc32c_hw(uint64_t, uint64_t, uint32_t) {
  // NOTE - HARDWARE CRC32 IS NOT CURRENTLY SUPPORTED
  // TODO - voznesenskym - Support hardware CRC32
  return 0;
}
static uint32_t gf_multiply_crc32_hw(uint64_t, uint64_t, uint32_t) {
  // NOTE - HARDWARE CRC32 IS NOT CURRENTLY SUPPORTED
  // TODO - voznesenskym - Support hardware CRC32
  return 0;
}

// #endif

static constexpr uint32_t crc32c_m = 0x82f63b78;
static constexpr uint32_t crc32_m = 0xedb88320;

/*
 * Pre-calculated powers tables for crc32c and crc32.
 */
static constexpr std::array<uint32_t, 62> const crc32c_powers =
    gf_powers_make<crc32c_m>{}(std::make_index_sequence<62>{});
static constexpr std::array<uint32_t, 62> const crc32_powers =
    gf_powers_make<crc32_m>{}(std::make_index_sequence<62>{});

template <typename F>
static uint32_t crc32_append_zeroes(
    F mult,
    uint32_t crc,
    size_t len,
    uint32_t polynomial,
    std::array<uint32_t, 62> const& powers_array) {
  auto powers = powers_array.data();

  // Append by multiplying by consecutive powers of two of the zeroes
  // array
  len >>= 2;

  while (len) {
    // Advance directly to next bit set.
    auto r = llvm::findFirstSet(len) - 1;
    len >>= r;
    powers += r;

    crc = mult(crc, *powers, polynomial);

    len >>= 1;
    powers++;
  }

  return crc;
}

namespace detail {

uint32_t crc32_combine_sw(uint32_t crc1, uint32_t crc2, size_t crc2len) {
  return crc2 ^
      crc32_append_zeroes(gf_multiply_sw, crc1, crc2len, crc32_m, crc32_powers);
}

uint32_t crc32_combine_hw(uint32_t crc1, uint32_t crc2, size_t crc2len) {
  return crc2 ^
      crc32_append_zeroes(
             gf_multiply_crc32_hw, crc1, crc2len, crc32_m, crc32_powers);
}

uint32_t crc32c_combine_sw(uint32_t crc1, uint32_t crc2, size_t crc2len) {
  return crc2 ^
      crc32_append_zeroes(
             gf_multiply_sw, crc1, crc2len, crc32c_m, crc32c_powers);
}

uint32_t crc32c_combine_hw(uint32_t crc1, uint32_t crc2, size_t crc2len) {
  return crc2 ^
      crc32_append_zeroes(
             gf_multiply_crc32c_hw, crc1, crc2len, crc32c_m, crc32c_powers);
}


uint32_t
crc32c_sw(const uint8_t* data, size_t nbytes, uint32_t startingChecksum);


// #if FOLLY_SSE_PREREQ(4, 2)

uint32_t
crc32_sw(const uint8_t* data, size_t nbytes, uint32_t startingChecksum);

#if FOLLY_SSE_PREREQ(4, 2)
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

bool crc32c_hw_supported() {
  static folly::CpuId id;
  return id.sse42();
}

bool crc32_hw_supported() {
  static folly::CpuId id;
  return id.sse42();
}

#else

uint32_t crc32_hw(
    const uint8_t* /* data */,
    size_t /* nbytes */,
    uint32_t /* startingChecksum */) {
  throw std::runtime_error("crc32_hw is not implemented on this platform");
}

bool crc32c_hw_supported() {
  return false;
}

bool crc32_hw_supported() {
  return false;
}
#endif


template <uint32_t CRC_POLYNOMIAL>
uint32_t crc_sw(const uint8_t* data, size_t nbytes, uint32_t startingChecksum) {
  // Reverse the bits in the starting checksum so they'll be in the
  // right internal format for Boost's CRC engine.
  //     O(1)-time, branchless bit reversal algorithm from
  //     http://graphics.stanford.edu/~seander/bithacks.html
  startingChecksum = ((startingChecksum >> 1) & 0x55555555) |
      ((startingChecksum & 0x55555555) << 1);
  startingChecksum = ((startingChecksum >> 2) & 0x33333333) |
      ((startingChecksum & 0x33333333) << 2);
  startingChecksum = ((startingChecksum >> 4) & 0x0f0f0f0f) |
      ((startingChecksum & 0x0f0f0f0f) << 4);
  startingChecksum = ((startingChecksum >> 8) & 0x00ff00ff) |
      ((startingChecksum & 0x00ff00ff) << 8);
  startingChecksum = (startingChecksum >> 16) | (startingChecksum << 16);

  boost::crc_optimal<32, CRC_POLYNOMIAL, ~0U, 0, true, true> sum(
      startingChecksum);
  sum.process_bytes(data, nbytes);
  return sum.checksum();
}

uint32_t
crc32c_sw(const uint8_t* data, size_t nbytes, uint32_t startingChecksum) {
  constexpr uint32_t CRC32C_POLYNOMIAL = 0x1EDC6F41;
  return crc_sw<CRC32C_POLYNOMIAL>(data, nbytes, startingChecksum);
}

uint32_t
crc32_sw(const uint8_t* data, size_t nbytes, uint32_t startingChecksum) {
  constexpr uint32_t CRC32_POLYNOMIAL = 0x04C11DB7;
  return crc_sw<CRC32_POLYNOMIAL>(data, nbytes, startingChecksum);
}


} // namespace detail

uint32_t crc32c(const uint8_t* data, size_t nbytes, uint32_t startingChecksum) {
    if (detail::crc32_hw_supported()) {
      return detail::crc32_hw(data, nbytes, startingChecksum);
    } else {
      return detail::crc32_sw(data, nbytes, startingChecksum);
    }
}

uint32_t crc32(const uint8_t* data, size_t nbytes, uint32_t startingChecksum) {
  if (detail::crc32_hw_supported()) {
    return detail::crc32_hw(data, nbytes, startingChecksum);
  } else {
    return detail::crc32_sw(data, nbytes, startingChecksum);
  }
}

uint32_t
crc32_type(const uint8_t* data, size_t nbytes, uint32_t startingChecksum) {
  return ~crc32(data, nbytes, startingChecksum);
}

uint32_t crc32_combine(uint32_t crc1, uint32_t crc2, size_t crc2len) {
  // Append up to 32 bits of zeroes in the normal way
  uint8_t data[4] = {0, 0, 0, 0};
  auto len = crc2len & 3;
  if (len) {
    crc1 = crc32(data, len, crc1);
  }

  if (detail::crc32_hw_supported()) {
    return detail::crc32_combine_hw(crc1, crc2, crc2len);
  } else {
    return detail::crc32_combine_sw(crc1, crc2, crc2len);
  }
}

uint32_t crc32c_combine(uint32_t crc1, uint32_t crc2, size_t crc2len) {
  // Append up to 32 bits of zeroes in the normal way
  uint8_t data[4] = {0, 0, 0, 0};
  auto len = crc2len & 3;
  if (len) {
    crc1 = crc32c(data, len, crc1);
  }

  if (detail::crc32_hw_supported()) {
    return detail::crc32c_combine_hw(crc1, crc2, crc2len - len);
  } else {
    return detail::crc32c_combine_sw(crc1, crc2, crc2len - len);
  }
}
}

extern "C" {
mz_ulong mz_crc32(mz_ulong crc, const mz_uint8* ptr, size_t buf_len) {
  // return 0;
  return detail::crc32(ptr, buf_len, crc);
};
}
