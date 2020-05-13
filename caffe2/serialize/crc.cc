#include "miniz.h"
#include <iostream>

#include "caffe2/serialize/boost_crc.h"
#include <algorithm>
#include <stdexcept>
#include <array>
#include <c10/util/llvmMathExtras.h>


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
// #endif

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
    return detail::crc32c_sw(data, nbytes, startingChecksum);
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
  return detail::crc32(ptr, crc, buf_len);
};
}
