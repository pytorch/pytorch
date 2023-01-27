#include <c10/util/BFloat16.h>
#include <c10/util/irange.h>
#include <torch/csrc/utils/byte_order.h>

#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#include <stdlib.h>
#endif

#if defined(__APPLE__) || defined(__FreeBSD__)
#include <machine/endian.h>
#elif !defined(_WIN32) && !defined(_WIN64)
#include <endian.h>
#endif

namespace {

static inline uint16_t decodeUInt16LE(const uint8_t* data) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint16_t output;
  memcpy(&output, data, sizeof(uint16_t));
  return output;
}

static inline uint16_t decodeUInt16BE(const uint8_t* data) {
  uint16_t output = decodeUInt16LE(data);
  swapBytes16(&output);
  return output;
}

static inline uint32_t decodeUInt32LE(const uint8_t* data) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t output;
  memcpy(&output, data, sizeof(uint32_t));
  return output;
}

static inline uint32_t decodeUInt32BE(const uint8_t* data) {
  uint32_t output = decodeUInt32LE(data);
  swapBytes32(&output);
  return output;
}

static inline uint64_t decodeUInt64LE(const uint8_t* data) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint64_t output;
  memcpy(&output, data, sizeof(uint64_t));
  return output;
}

static inline uint64_t decodeUInt64BE(const uint8_t* data) {
  uint64_t output = decodeUInt64LE(data);
  swapBytes64(&output);
  return output;
}

} // anonymous namespace

namespace torch {
namespace utils {

THPByteOrder THP_nativeByteOrder() {
#if defined(_WIN32) || defined(_WIN64)
  return THP_LITTLE_ENDIAN;
#elif defined(__BYTE_ORDER__)
  #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return THP_LITTLE_ENDIAN;
  #elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return THP_BIG_ENDIAN;
  #else
    uint32_t x = 1;
    return *(uint8_t*)&x ? THP_LITTLE_ENDIAN : THP_BIG_ENDIAN;
  #endif
#else
  uint32_t x = 1;
  return *(uint8_t*)&x ? THP_LITTLE_ENDIAN : THP_BIG_ENDIAN;
#endif
}

void THP_decodeInt16Buffer(
    int16_t* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    dst[i] =
        (int16_t)(do_byte_swap ? decodeUInt16BE(src) : decodeUInt16LE(src));
    src += sizeof(int16_t);
  }
}

void THP_decodeInt32Buffer(
    int32_t* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    dst[i] =
        (int32_t)(do_byte_swap ? decodeUInt32BE(src) : decodeUInt32LE(src));
    src += sizeof(int32_t);
  }
}

void THP_decodeInt64Buffer(
    int64_t* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    dst[i] =
        (int64_t)(do_byte_swap ? decodeUInt64BE(src) : decodeUInt64LE(src));
    src += sizeof(int64_t);
  }
}

void THP_decodeHalfBuffer(
    c10::Half* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint16_t x;
      c10::Half f;
    };
    x = (do_byte_swap ? decodeUInt16BE(src) : decodeUInt16LE(src));
    dst[i] = f;
    src += sizeof(uint16_t);
  }
}

void THP_decodeBFloat16Buffer(
    at::BFloat16* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    uint16_t x = (do_byte_swap ? decodeUInt16BE(src) : decodeUInt16LE(src));
    std::memcpy(&dst[i], &x, sizeof(dst[i]));
    src += sizeof(uint16_t);
  }
}

void THP_decodeBoolBuffer(
    bool* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    dst[i] = (int)src[i] != 0 ? true : false;
  }
}

void THP_decodeFloatBuffer(
    float* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t x;
      float f;
    };
    x = (do_byte_swap ? decodeUInt32BE(src) : decodeUInt32LE(src));
    dst[i] = f;
    src += sizeof(float);
  }
}

void THP_decodeDoubleBuffer(
    double* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint64_t x;
      double d;
    };
    x = (do_byte_swap ? decodeUInt64BE(src) : decodeUInt64LE(src));
    dst[i] = d;
    src += sizeof(double);
  }
}

void THP_decodeComplexFloatBuffer(
    c10::complex<float>* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t x;
      float re;
    };
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t y;
      float im;
    };

    x = (do_byte_swap ? decodeUInt32BE(src) : decodeUInt32LE(src));
    src += sizeof(float);
    y = (do_byte_swap ? decodeUInt32BE(src) : decodeUInt32LE(src));
    src += sizeof(float);

    dst[i] = c10::complex<float>(re, im);
  }
}

void THP_decodeComplexDoubleBuffer(
    c10::complex<double>* dst,
    const uint8_t* src,
    bool do_byte_swap,
    size_t len) {
  for (const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t x;
      double re;
    };
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union {
      uint32_t y;
      double im;
    };

    x = (do_byte_swap ? decodeUInt64BE(src) : decodeUInt64LE(src));
    src += sizeof(double);
    y = (do_byte_swap ? decodeUInt64BE(src) : decodeUInt64LE(src));
    src += sizeof(double);

    dst[i] = c10::complex<double>(re, im);
  }
}

void THP_decodeInt16Buffer(
    int16_t* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len) {
  THP_decodeInt16Buffer(dst, src, (order == THP_BIG_ENDIAN), len);
}

void THP_decodeInt32Buffer(
    int32_t* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len) {
  THP_decodeInt32Buffer(dst, src, (order == THP_BIG_ENDIAN), len);
}

void THP_decodeInt64Buffer(
    int64_t* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len) {
  THP_decodeInt64Buffer(dst, src, (order == THP_BIG_ENDIAN), len);
}

void THP_decodeHalfBuffer(
    c10::Half* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len) {
  THP_decodeHalfBuffer(dst, src, (order == THP_BIG_ENDIAN), len);
}

void THP_decodeBFloat16Buffer(
    at::BFloat16* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len) {
  THP_decodeBFloat16Buffer(dst, src, (order == THP_BIG_ENDIAN), len);
}

void THP_decodeBoolBuffer(
    bool* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len) {
  THP_decodeBoolBuffer(dst, src, (order == THP_BIG_ENDIAN), len);
}

void THP_decodeFloatBuffer(
    float* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len) {
  THP_decodeFloatBuffer(dst, src, (order == THP_BIG_ENDIAN), len);
}

void THP_decodeDoubleBuffer(
    double* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len) {
  THP_decodeDoubleBuffer(dst, src, (order == THP_BIG_ENDIAN), len);
}

void THP_decodeComplexFloatBuffer(
    c10::complex<float>* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len) {
  THP_decodeComplexFloatBuffer(dst, src, (order == THP_BIG_ENDIAN), len);
}

void THP_decodeComplexDoubleBuffer(
    c10::complex<double>* dst,
    const uint8_t* src,
    THPByteOrder order,
    size_t len) {
  THP_decodeComplexDoubleBuffer(dst, src, (order == THP_BIG_ENDIAN), len);
}

void THP_encodeInt16Buffer(
    uint8_t* dst,
    const int16_t* src,
    THPByteOrder order,
    size_t len) {
  memcpy(dst, src, sizeof(int16_t) * len);
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(len)) {
      (void)i;
      swapBytes16(dst);
      dst += sizeof(int16_t);
    }
  }
}

void THP_encodeInt32Buffer(
    uint8_t* dst,
    const int32_t* src,
    THPByteOrder order,
    size_t len) {
  memcpy(dst, src, sizeof(int32_t) * len);
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(len)) {
      (void)i;
      swapBytes32(dst);
      dst += sizeof(int32_t);
    }
  }
}

void THP_encodeInt64Buffer(
    uint8_t* dst,
    const int64_t* src,
    THPByteOrder order,
    size_t len) {
  memcpy(dst, src, sizeof(int64_t) * len);
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(len)) {
      (void)i;
      swapBytes64(dst);
      dst += sizeof(int64_t);
    }
  }
}

void THP_encodeFloatBuffer(
    uint8_t* dst,
    const float* src,
    THPByteOrder order,
    size_t len) {
  memcpy(dst, src, sizeof(float) * len);
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(len)) {
      (void)i;
      swapBytes32(dst);
      dst += sizeof(float);
    }
  }
}

void THP_encodeDoubleBuffer(
    uint8_t* dst,
    const double* src,
    THPByteOrder order,
    size_t len) {
  memcpy(dst, src, sizeof(double) * len);
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(len)) {
      (void)i;
      swapBytes64(dst);
      dst += sizeof(double);
    }
  }
}

template <typename T>
std::vector<T> complex_to_float(const c10::complex<T>* src, size_t len) {
  std::vector<T> new_src;
  new_src.reserve(2 * len);
  for (const auto i : c10::irange(len)) {
    auto elem = src[i];
    new_src.emplace_back(elem.real());
    new_src.emplace_back(elem.imag());
  }
  return new_src;
}

void THP_encodeComplexFloatBuffer(
    uint8_t* dst,
    const c10::complex<float>* src,
    THPByteOrder order,
    size_t len) {
  auto new_src = complex_to_float(src, len);
  memcpy(dst, static_cast<void*>(&new_src), 2 * sizeof(float) * len);
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(2 * len)) {
      (void)i; // Suppress unused variable warning
      swapBytes32(dst);
      dst += sizeof(float);
    }
  }
}

void THP_encodeCompelxDoubleBuffer(
    uint8_t* dst,
    const c10::complex<double>* src,
    THPByteOrder order,
    size_t len) {
  auto new_src = complex_to_float(src, len);
  memcpy(dst, static_cast<void*>(&new_src), 2 * sizeof(double) * len);
  if (order != THP_nativeByteOrder()) {
    for (const auto i : c10::irange(2 * len)) {
      (void)i; // Suppress unused variable warning
      swapBytes64(dst);
      dst += sizeof(double);
    }
  }
}

} // namespace utils
} // namespace torch
