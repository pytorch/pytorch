#pragma once

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>

namespace at {
namespace vec256 {
namespace {

#ifdef __AVX2__

template <>
struct Vec256<uint8_t> {
private:
  __m256i values;
  static inline __m256i invert(const __m256i& v) {
    const auto ones = _mm256_set1_epi64x(-1);
    return _mm256_xor_si256(ones, v);
  }
public:
  static constexpr int size() {
    return 32;
  }

  Vec256() {}
  Vec256(__m256i v) : values(v) {}
  operator __m256i() const {
    return values;
  }
  Vec256(uint8_t v) { values = _mm256_set1_epi8(v); }
  Vec256(uint8_t val1, uint8_t val2,
         uint8_t val3, uint8_t val4,
         uint8_t val5, uint8_t val6,
         uint8_t val7, uint8_t val8,
         uint8_t val9, uint8_t val10,
         uint8_t val11, uint8_t val12,
         uint8_t val13, uint8_t val14,
         uint8_t val15, uint8_t val16,
         uint8_t val17, uint8_t val18,
         uint8_t val19, uint8_t val20,
         uint8_t val21, uint8_t val22,
         uint8_t val23, uint8_t val24,
         uint8_t val25, uint8_t val26,
         uint8_t val27, uint8_t val28,
         uint8_t val29, uint8_t val30,
         uint8_t val31, uint8_t val32) {
    values = _mm256_setr_epi8(
      val1,  val2,  val3,  val4,  val5,  val6,  val7,  val8,
      val9,  val10, val11, val12, val13, val14, val15, val16,
      val17, val18, val19, val20, val21, val22, val23, val24,
      val25, val26, val27, val28, val29, val30, val31, val32);
  }
  template <int64_t mask>
  static Vec256<uint8_t> blend(
      Vec256<uint8_t> a, Vec256<uint8_t> b) {
    __at_align32__ uint8_t tmp_values[size()];
    a.store(tmp_values);
    if (mask & 0x01)
      tmp_values[0] = _mm256_extract_epi8(b.values, 0);
    if (mask & 0x02)
      tmp_values[1] = _mm256_extract_epi8(b.values, 1);
    if (mask & 0x04)
      tmp_values[2] = _mm256_extract_epi8(b.values, 2);
    if (mask & 0x08)
      tmp_values[3] = _mm256_extract_epi8(b.values, 3);
    if (mask & 0x10)
      tmp_values[4] = _mm256_extract_epi8(b.values, 4);
    if (mask & 0x20)
      tmp_values[5] = _mm256_extract_epi8(b.values, 5);
    if (mask & 0x40)
      tmp_values[6] = _mm256_extract_epi8(b.values, 6);
    if (mask & 0x80)
      tmp_values[7] = _mm256_extract_epi8(b.values, 7);
    if (mask & 0x100)
      tmp_values[8] = _mm256_extract_epi8(b.values, 8);
    if (mask & 0x200)
      tmp_values[9] = _mm256_extract_epi8(b.values, 9);
    if (mask & 0x400)
      tmp_values[10] = _mm256_extract_epi8(b.values, 10);
    if (mask & 0x800)
      tmp_values[11] = _mm256_extract_epi8(b.values, 11);
    if (mask & 0x1000)
      tmp_values[12] = _mm256_extract_epi8(b.values, 12);
    if (mask & 0x2000)
      tmp_values[13] = _mm256_extract_epi8(b.values, 13);
    if (mask & 0x4000)
      tmp_values[14] = _mm256_extract_epi8(b.values, 14);
    if (mask & 0x8000)
      tmp_values[15] = _mm256_extract_epi8(b.values, 15);
    if (mask & 0x10000)
      tmp_values[16] = _mm256_extract_epi8(b.values, 16);
    if (mask & 0x20000)
      tmp_values[17] = _mm256_extract_epi8(b.values, 17);
    if (mask & 0x40000)
      tmp_values[18] = _mm256_extract_epi8(b.values, 18);
    if (mask & 0x80000)
      tmp_values[19] = _mm256_extract_epi8(b.values, 19);
    if (mask & 0x100000)
      tmp_values[20] = _mm256_extract_epi8(b.values, 20);
    if (mask & 0x200000)
      tmp_values[21] = _mm256_extract_epi8(b.values, 21);
    if (mask & 0x400000)
      tmp_values[22] = _mm256_extract_epi8(b.values, 22);
    if (mask & 0x800000)
      tmp_values[23] = _mm256_extract_epi8(b.values, 23);
    if (mask & 0x1000000)
      tmp_values[24] = _mm256_extract_epi8(b.values, 24);
    if (mask & 0x2000000)
      tmp_values[25] = _mm256_extract_epi8(b.values, 25);
    if (mask & 0x4000000)
      tmp_values[26] = _mm256_extract_epi8(b.values, 26);
    if (mask & 0x8000000)
      tmp_values[27] = _mm256_extract_epi8(b.values, 27);
    if (mask & 0x10000000)
      tmp_values[28] = _mm256_extract_epi8(b.values, 28);
    if (mask & 0x20000000)
      tmp_values[29] = _mm256_extract_epi8(b.values, 29);
    if (mask & 0x40000000)
      tmp_values[30] = _mm256_extract_epi8(b.values, 30);
    if (mask & 0x80000000LL)
      tmp_values[31] = _mm256_extract_epi8(b.values, 31);
    return loadu(tmp_values);
  }
  static Vec256<uint8_t> blendv(
      const Vec256<uint8_t>& a,
      const Vec256<uint8_t>& b,
      const Vec256<uint8_t>& mask) {
    return _mm256_blendv_epi8(a.values, b.values, mask.values);
  }
  static Vec256<uint8_t> arange(
      uint8_t base = 0, uint8_t step = 1) {
    return Vec256<uint8_t>(
      base,             base +      step, base +  2 * step, base +  3 * step,
      base +  4 * step, base +  5 * step, base +  6 * step, base +  7 * step,
      base +  8 * step, base +  9 * step, base + 10 * step, base + 11 * step,
      base + 12 * step, base + 13 * step, base + 14 * step, base + 15 * step,
      base + 16 * step, base + 17 * step, base + 18 * step, base + 19 * step,
      base + 20 * step, base + 21 * step, base + 22 * step, base + 23 * step,
      base + 24 * step, base + 25 * step, base + 26 * step, base + 27 * step,
      base + 28 * step, base + 29 * step, base + 30 * step, base + 31 * step);
  }
  static Vec256<uint8_t>
  set(Vec256<uint8_t> a,
      Vec256<uint8_t> b,
      uint8_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
      case 4:
        return blend<15>(a, b);
      case 5:
        return blend<31>(a, b);
      case 6:
        return blend<63>(a, b);
      case 7:
        return blend<127>(a, b);
      case 8:
        return blend<255>(a, b);
      case 9:
        return blend<511>(a, b);
      case 10:
        return blend<1023>(a, b);
      case 11:
        return blend<2047>(a, b);
      case 12:
        return blend<4095>(a, b);
      case 13:
        return blend<8191>(a, b);
      case 14:
        return blend<16383>(a, b);
      case 15:
        return blend<32767>(a, b);
      case 16:
        return blend<65535>(a, b);
      case 17:
        return blend<131071>(a, b);
      case 18:
        return blend<262143>(a, b);
      case 19:
        return blend<524287>(a, b);
      case 20:
        return blend<1048575>(a, b);
      case 21:
        return blend<2097151>(a, b);
      case 22:
        return blend<4194303>(a, b);
      case 23:
        return blend<8388607>(a, b);
      case 24:
        return blend<16777215>(a, b);
      case 25:
        return blend<33554431>(a, b);
      case 26:
        return blend<67108863>(a, b);
      case 27:
        return blend<134217727>(a, b);
      case 28:
        return blend<268435455>(a, b);
      case 29:
        return blend<536870911>(a, b);
      case 30:
        return blend<1073741823>(a, b);
      case 31:
        return blend<2147483647>(a, b);
    }
    return b;
  }
  static Vec256<uint8_t> loadu(const void* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
  static Vec256<uint8_t> loadu(const void* ptr, uint8_t count) {
    __at_align32__ uint8_t tmp_values[size()];
    std::memcpy(tmp_values, ptr, count * sizeof(uint8_t));
    return loadu(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), values);
    } else if (count > 0) {
      __at_align32__ uint8_t tmp_values[size()];
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(uint8_t));
    }
  }
  const uint8_t& operator[](int idx) const  = delete;
  uint8_t& operator[](int idx)  = delete;
  Vec256<uint8_t> abs() const {
    return values;
  }
  Vec256<uint8_t> operator==(const Vec256<uint8_t>& other) const {
    return _mm256_cmpeq_epi8(values, other.values);
  }
  Vec256<uint8_t> operator!=(const Vec256<uint8_t>& other) const {
    return invert(_mm256_cmpeq_epi8(values, other.values));
  }
  Vec256<uint8_t> operator<(const Vec256<uint8_t>& other) const {
    // cannot use _mm256_cmpgt_epi8 which is for signed bytes
    auto min = _mm256_min_epu8(values, other.values);
    return invert(_mm256_cmpeq_epi8(min, other.values));
  }
  Vec256<uint8_t> operator<=(const Vec256<uint8_t>& other) const {
    // cannot use _mm256_cmpgt_epi8 which is for signed bytes
    return _mm256_cmpeq_epi8(_mm256_min_epu8(values, other.values), values);
  }
  Vec256<uint8_t> operator>(const Vec256<uint8_t>& other) const {
    // cannot use _mm256_cmpgt_epi8 which is for signed bytes
    auto max = _mm256_max_epu8(values, other.values);
    return invert(_mm256_cmpeq_epi8(max, other.values));
  }
  Vec256<uint8_t> operator>=(const Vec256<uint8_t>& other) const {
    // cannot use _mm256_cmpgt_epi8 which is for signed bytes
    return _mm256_cmpeq_epi8(_mm256_max_epu8(values, other.values), values);
  }
};

template <>
Vec256<uint8_t> inline operator+(
    const Vec256<uint8_t>& a, const Vec256<uint8_t>& b) {
  return _mm256_add_epi8(a, b);
}

template <>
Vec256<uint8_t> inline operator-(
    const Vec256<uint8_t>& a, const Vec256<uint8_t>& b) {
  return _mm256_sub_epi8(a, b);
}

#define DEFINE_BOOL_ARITHMETIC_FUNC(op)                                     \
template <>                                                                 \
Vec256<uint8_t> inline operator op(                                         \
    const Vec256<uint8_t>& a, const Vec256<uint8_t>& b) {                   \
  uint8_t values_a[Vec256<uint8_t>::size()];                                \
  uint8_t values_b[Vec256<uint8_t>::size()];                                \
  a.store(values_a);                                                        \
  b.store(values_b);                                                        \
  for (int i = 0; i != Vec256<uint8_t>::size(); i++) {                      \
    values_a[i] = values_a[i] op values_b[i];                               \
  }                                                                         \
  return Vec256<uint8_t>::loadu(values_a);                                  \
}                                                                           \

DEFINE_BOOL_ARITHMETIC_FUNC(*)
DEFINE_BOOL_ARITHMETIC_FUNC(/)

#undef DEFINE_BOOL_ARITHMETIC_FUNC

#define DEFINE_BOOL_BINARY_OP(op, func)                                     \
template <>                                                                 \
Vec256<uint8_t> inline operator op(                                         \
    const Vec256<uint8_t>& a, const Vec256<uint8_t>& b) {                   \
  return func(a, b);                                                        \
}

DEFINE_BOOL_BINARY_OP(&, _mm256_and_si256)
DEFINE_BOOL_BINARY_OP(|, _mm256_or_si256)
DEFINE_BOOL_BINARY_OP(^, _mm256_xor_si256)

#undef DEFINE_BOOL_BINARY_OP

template <>
Vec256<uint8_t> inline operator&&(
    const Vec256<uint8_t> &a, const Vec256<uint8_t> &b) {
  static auto zero = _mm256_set1_epi64x(0);
  static auto one  = _mm256_set1_epi8(1);
  // _mm256_cmpeq_epi8 set all bits in byte to 1 if the two values equal. We
  // only use the lowest bit in every byte to make sure it returns 1 instead of
  // 255.
  auto ma = _mm256_xor_si256(one, _mm256_cmpeq_epi8(__m256i(a), zero));
  auto mb = _mm256_xor_si256(one, _mm256_cmpeq_epi8(__m256i(b), zero));
  return _mm256_and_si256(one, _mm256_and_si256(ma, mb));
}

#endif

}}}
