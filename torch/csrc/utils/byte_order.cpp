#include <torch/csrc/utils/byte_order.h>
#include <c10/util/BFloat16.h>
#include <c10/util/irange.h>

#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#include <stdlib.h>
#endif

namespace {

static inline void swapBytes16(void *ptr)
{
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint16_t output;
  memcpy(&output, ptr, sizeof(uint16_t));
#if defined(_MSC_VER) && !defined(_DEBUG)
  output = _byteswap_ushort(output);
#elif defined(__llvm__) || defined(__GNUC__) && !defined(__ICC)
  output = __builtin_bswap16(output);
#else
  uint16_t Hi = output >> 8;
  uint16_t Lo = output << 8;
  output = Hi | Lo;
#endif
  memcpy(ptr, &output, sizeof(uint16_t));
}

static inline void swapBytes32(void *ptr)
{
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t output;
  memcpy(&output, ptr, sizeof(uint32_t));
#if defined(_MSC_VER) && !defined(_DEBUG)
  output = _byteswap_ulong(output);
#elif defined(__llvm__) || defined(__GNUC__) && !defined(__ICC)
  output = __builtin_bswap32(output);
#else
   uint32_t Byte0 = output & 0x000000FF;
   uint32_t Byte1 = output & 0x0000FF00;
   uint32_t Byte2 = output & 0x00FF0000;
   uint32_t Byte3 = output & 0xFF000000;
   output = (Byte0 << 24) | (Byte1 << 8) | (Byte2 >> 8) | (Byte3 >> 24);
#endif
  memcpy(ptr, &output, sizeof(uint32_t));
}

static inline void swapBytes64(void *ptr)
{
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint64_t output;
  memcpy(&output, ptr, sizeof(uint64_t));
#if defined(_MSC_VER)
  output = _byteswap_uint64(output);
#elif defined(__llvm__) || defined(__GNUC__) && !defined(__ICC)
  output = __builtin_bswap64(output);
#else
  uint64_t Byte0 = output & 0x00000000000000FF;
  uint64_t Byte1 = output & 0x000000000000FF00;
  uint64_t Byte2 = output & 0x0000000000FF0000;
  uint64_t Byte3 = output & 0x00000000FF000000;
  uint64_t Byte4 = output & 0x000000FF00000000;
  uint64_t Byte5 = output & 0x0000FF0000000000;
  uint64_t Byte6 = output & 0x00FF000000000000;
  uint64_t Byte7 = output & 0xFF00000000000000;
  output = (Byte0 << (7*8)) | (Byte1 << (5*8)) | (Byte2 << (3*8)) | (Byte3 << (1*8)) |
           (Byte7 >> (7*8)) | (Byte6 >> (5*8)) | (Byte5 >> (3*8)) | (Byte4 >> (1*8));
#endif
  memcpy(ptr, &output, sizeof(uint64_t));
}

static inline uint16_t decodeUInt16LE(const uint8_t *data) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint16_t output;
  memcpy(&output, data, sizeof(uint16_t));
  return output;
}

static inline uint16_t decodeUInt16BE(const uint8_t *data) {
  uint16_t output = decodeUInt16LE(data);
  swapBytes16(&output);
  return output;
}

static inline uint32_t decodeUInt32LE(const uint8_t *data) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t output;
  memcpy(&output, data, sizeof(uint32_t));
  return output;
}

static inline uint32_t decodeUInt32BE(const uint8_t *data) {
  uint32_t output = decodeUInt32LE(data);
  swapBytes32(&output);
  return output;
}

static inline uint64_t decodeUInt64LE(const uint8_t *data) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint64_t output;
  memcpy(&output, data, sizeof(uint64_t));
  return output;
}

static inline uint64_t decodeUInt64BE(const uint8_t *data) {
  uint64_t output = decodeUInt64LE(data);
  swapBytes64(&output);
  return output;
}

} // anonymous namespace

namespace torch {
namespace utils {

THPByteOrder THP_nativeByteOrder()
{
  uint32_t x = 1;
  return *(uint8_t*)&x ? THP_LITTLE_ENDIAN : THP_BIG_ENDIAN;
}

void THP_decodeInt16Buffer(int16_t* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for(const auto i : c10::irange(len)) {
    dst[i] = (int16_t)(
        order == THP_BIG_ENDIAN ? decodeUInt16BE(src) : decodeUInt16LE(src));
    src += sizeof(int16_t);
  }
}

void THP_decodeInt32Buffer(int32_t* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for(const auto i : c10::irange(len)) {
    dst[i] = (int32_t)(
        order == THP_BIG_ENDIAN ? decodeUInt32BE(src) : decodeUInt32LE(src));
    src += sizeof(int32_t);
  }
}

void THP_decodeInt64Buffer(int64_t* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for(const auto i : c10::irange(len)) {
    dst[i] = (int64_t)(
        order == THP_BIG_ENDIAN ? decodeUInt64BE(src) : decodeUInt64LE(src));
    src += sizeof(int64_t);
  }
}

void THP_decodeHalfBuffer(THHalf* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for(const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union { uint16_t x; THHalf f; };
    x = (order == THP_BIG_ENDIAN ? decodeUInt16BE(src) : decodeUInt16LE(src));
    dst[i] = f;
    src += sizeof(uint16_t);
  }
}

void THP_decodeBFloat16Buffer(at::BFloat16* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for(const auto i : c10::irange(len)) {
    uint16_t x =
        (order == THP_BIG_ENDIAN ? decodeUInt16BE(src) : decodeUInt16LE(src));
    std::memcpy(&dst[i], &x, sizeof(dst[i]));
    src += sizeof(uint16_t);
  }
}

void THP_decodeBoolBuffer(bool* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for(const auto i : c10::irange(len)) {
    dst[i] = (int)src[i] != 0 ? true : false;
  }
}

void THP_decodeFloatBuffer(float* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for(const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union { uint32_t x; float f; };
    x = (order == THP_BIG_ENDIAN ? decodeUInt32BE(src) : decodeUInt32LE(src));
    dst[i] = f;
    src += sizeof(float);
  }
}

void THP_decodeDoubleBuffer(double* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for(const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union { uint64_t x; double d; };
    x = (order == THP_BIG_ENDIAN ? decodeUInt64BE(src) : decodeUInt64LE(src));
    dst[i] = d;
    src += sizeof(double);
  }
}

void THP_decodeComplexFloatBuffer(c10::complex<float>* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for(const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union { uint32_t x; float re; };
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union { uint32_t y; float im; };

    x = (order == THP_BIG_ENDIAN ? decodeUInt32BE(src) : decodeUInt32LE(src));
    src += sizeof(float);
    y = (order == THP_BIG_ENDIAN ? decodeUInt32BE(src) : decodeUInt32LE(src));
    src += sizeof(float);

    dst[i] = c10::complex<float>(re, im);
  }
}

void THP_decodeComplexDoubleBuffer(c10::complex<double>* dst, const uint8_t* src, THPByteOrder order, size_t len)
{
  for(const auto i : c10::irange(len)) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union { uint32_t x; double re; };
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    union { uint32_t y; double im; };

    x = (order == THP_BIG_ENDIAN ? decodeUInt64BE(src) : decodeUInt64LE(src));
    src += sizeof(double);
    y = (order == THP_BIG_ENDIAN ? decodeUInt64BE(src) : decodeUInt64LE(src));
    src += sizeof(double);

    dst[i] = c10::complex<double>(re, im);
  }
}

void THP_encodeInt16Buffer(uint8_t* dst, const int16_t* src, THPByteOrder order, size_t len)
{
  memcpy(dst, src, sizeof(int16_t) * len);
  if (order != THP_nativeByteOrder()) {
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores,clang-diagnostic-unused-variable)
    for(const auto i : c10::irange(len)) {
      swapBytes16(dst);
      dst += sizeof(int16_t);
    }
  }
}

void THP_encodeInt32Buffer(uint8_t* dst, const int32_t* src, THPByteOrder order, size_t len)
{
  memcpy(dst, src, sizeof(int32_t) * len);
  if (order != THP_nativeByteOrder()) {
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores,clang-diagnostic-unused-variable)
    for(const auto i : c10::irange(len)) {
      swapBytes32(dst);
      dst += sizeof(int32_t);
    }
  }
}

void THP_encodeInt64Buffer(uint8_t* dst, const int64_t* src, THPByteOrder order, size_t len)
{
  memcpy(dst, src, sizeof(int64_t) * len);
  if (order != THP_nativeByteOrder()) {
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores,clang-diagnostic-unused-variable)
    for(const auto i : c10::irange(len)) {
      swapBytes64(dst);
      dst += sizeof(int64_t);
    }
  }
}

void THP_encodeFloatBuffer(uint8_t* dst, const float* src, THPByteOrder order, size_t len)
{
  memcpy(dst, src, sizeof(float) * len);
  if (order != THP_nativeByteOrder()) {
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores,clang-diagnostic-unused-variable)
    for(const auto i : c10::irange(len)) {
      swapBytes32(dst);
      dst += sizeof(float);
    }
  }
}

void THP_encodeDoubleBuffer(uint8_t* dst, const double* src, THPByteOrder order, size_t len)
{
  memcpy(dst, src, sizeof(double) * len);
  if (order != THP_nativeByteOrder()) {
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores,clang-diagnostic-unused-variable)
    for(const auto i : c10::irange(len)) {
      swapBytes64(dst);
      dst += sizeof(double);
    }
  }
}

template <typename T>
std::vector<T> complex_to_float(const c10::complex<T>* src, size_t len) {
  std::vector<T> new_src;
  new_src.reserve(2 * len);
  for(const auto i : c10::irange(len)) {
    auto elem = src[i];
    new_src.emplace_back(elem.real());
    new_src.emplace_back(elem.imag());
  }
  return new_src;
}

void THP_encodeComplexFloatBuffer(uint8_t* dst, const c10::complex<float>* src, THPByteOrder order, size_t len)
{
  auto new_src = complex_to_float(src, len);
  memcpy(dst, static_cast<void*>(&new_src), 2 * sizeof(float) * len);
  if (order != THP_nativeByteOrder()) {
    for (size_t i = 0; i < (2 * len); i++) {
      swapBytes32(dst);
      dst += sizeof(float);
    }
  }
}

void THP_encodeCompelxDoubleBuffer(uint8_t* dst, const c10::complex<double>* src, THPByteOrder order, size_t len)
{
  auto new_src = complex_to_float(src, len);
  memcpy(dst, static_cast<void*>(&new_src), 2 * sizeof(double) * len);
  if (order != THP_nativeByteOrder()) {
    for (size_t i = 0; i < (2 * len); i++) {
      swapBytes64(dst);
      dst += sizeof(double);
    }
  }
}

} // namespace utils
} // namespace torch
